"""Operations related to measurement sets
"""
from pathlib import Path 
from typing import NamedTuple, Optional

class MS(NamedTuple):
    """Helper to keep tracked of measurement set information
    """
    path: Path 
    column: Optional[str] = None 



