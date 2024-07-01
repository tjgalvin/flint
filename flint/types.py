from __future__ import (  # Used for mypy/pylance to like the return type of MS.with_options
    annotations,
)

from typing import NamedTuple, Collection


# TODO: Perhaps move these to flint.naming, and can be built up
# based on rules, e.g. imager used, source finder etc.
DEFAULT_TAR_RE_PATTERNS = (
    r".*MFS.*image\.fits",
    r".*linmos.*",
    r".*weight\.fits",
    r".*yaml",
    r".*\.txt",
    r".*png",
    r".*beam[0-9]+\.ms\.zip",
    r".*beam[0-9]+\.ms",
    r".*\.caltable",
    r".*\.tar",
    r".*\.csv",
)
DEFAULT_COPY_RE_PATTERNS = (r".*linmos.*fits", r".*weight\.fits", r".*png", r".*csv")


class ArchiveOptions(NamedTuple):
    """Container for options related to archiving products from flint workflows"""

    tar_file_re_patterns: Collection[str] = DEFAULT_TAR_RE_PATTERNS
    """Regular-expressions to use to collect files that should be tarballed"""
    copy_file_re_patterns: Collection[str] = DEFAULT_COPY_RE_PATTERNS
    """Regular-expressions used to identify files to copy into a final location (not tarred)"""

    def with_options(self, **kwargs) -> ArchiveOptions:
        opts = self._asdict()
        opts.update(**kwargs)

        return ArchiveOptions(**opts)
