# Installation

Provided an appropriate environment installation should be as simple as a
`pip install`.

However, on some systems there are interactions with `casacore` and building
`python-casacore` appropriately. Issues have been noted when interacting with
large measurement sets across components with different `casacore` versions.
This seems to happen even across container boundaries (i.e. different versions
in containers might play a role). The exact cause is not at all understood, but
it appears to be related to the version of `python-casacore`, `numpy` and
whether pre-built wheels are used.

In practise it might be easier to leverage `conda` to install the appropriate
`boost` and `casacore` libraries.

We have split out the `pip` dependencies that rely on `python-casacore`. These
can be installed by running:

```bash
pip install .[casa]
```

A helpful script below may be of use.

```bash
BRANCH="main" # replace this with appropriate branch or tag
DIR="flint_${BRANCH}"
PYVERSION="3.12"

mkdir "${DIR}" || exit
cd "${DIR}" || exit


git clone git@github.com:tjgalvin/flint.git && \
        cd flint && \
        git checkout "${BRANCH}"

conda create -y  -n "${DIR}" python="${PYVERSION}" &&  \
        source /home/$(whoami)/.bashrc && \
        conda activate "${DIR}" && \
        conda install -y -c conda-forge boost casacore && \
        PIP_NO_BINARY="python-casacore" pip install -e .[casa]
```

This may set up an appropriate environment that is compatible with the
containers currently being used.

:::{attention} For the moment there are no 'supported' container packaged within
this repository -- sorry! :::

See [containers](#containers) for more information.
