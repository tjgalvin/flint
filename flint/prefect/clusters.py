"""Some utility functions around the creation of Prefect task funners.

For this work we will be using Dask backed workers to perform the compute
operations.
"""

from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pkg_resources
import yaml
from prefect_dask import DaskTaskRunner


def list_packaged_clusters() -> List[str]:
    """Return a list of cluster names that are available in the packaged set of
    dask_jobqueue specification YAML files.

    Returns:
        list[str]: A list of preinstalled dask_jobqueue cluster specification files
    """
    yaml_files_dir = pkg_resources.resource_filename("aces", "cluster_configs/")
    yaml_files = glob(f"{yaml_files_dir}/*yaml")

    clusters = [Path(f).stem for f in yaml_files]

    return clusters


def get_cluster_spec(cluster: Union[str, Path]) -> Dict[Any, Any]:
    """
    Given a cluster name, obtain the appropriate SLURM configuration
    file appropriate for use with SLURMCluster.

    This cluster spec is expected to be consistent with the cluster_class
    and cluster_kwargs parameters that are used by dask_jobqueue based
    specifications.

    Args:
        cluster (Union[str,Path]): Name of cluster or path to a configuration to look up for processing

    Raises:
        ValueError: Raised when cluster is not in KNOWN_CLUSTERS and has not corresponding YAML file.

    Returns:
        dict[Any, Any]: Dictionary of know options/parameters for dask_jobqueue.SLURMCluster
    """

    KNOWN_CLUSTERS = ("galaxy",)
    yaml_file = None

    if Path(cluster).exists():
        yaml_file = cluster
    elif cluster == "galaxy":
        yaml_file = pkg_resources.resource_filename(
            "aces", "cluster_configs/galaxy_small.yaml"
        )
    else:
        yaml_file = pkg_resources.resource_filename(
            "aces", f"cluster_configs/{cluster}.yaml"
        )

    if yaml_file is None or not Path(yaml_file).exists():
        raise ValueError(
            f"{cluster=} is not known, or its YAML file could not be loaded. Known clusters are {KNOWN_CLUSTERS}"
        )

    with open(yaml_file, "r") as in_file:
        spec = yaml.load(in_file, Loader=yaml.Loader)

    return spec


def get_dask_runner(
    cluster: Union[str, Path] = "galaxy_small",
    extra_cluster_kwargs: Optional[Dict[str, Any]] = None,
) -> DaskTaskRunner:
    """Creates and returns a DaskTaskRunner configured to establised a SLURMCluster instance
    to manage a set of dask-workers. The SLURMCluster is currently configured only for Galaxy.

    Keyword Args:
        cluster (Union[str,Path]): The cluster name that will be used to search for a cluster specification file.
                       This could be the name of a known cluster, or the name of a yaml file installed
                       amoung the `cluster_configs` directory of the aces module.

    Returns:
        DaskTaskRunner: A dask task runner capable of being used as a task_runner for a prefect flow
    """

    spec = get_cluster_spec(cluster)

    if extra_cluster_kwargs is not None:
        spec["cluster_kwargs"].update(extra_cluster_kwargs)

    task_runner = DaskTaskRunner(**spec)

    return task_runner
