"""FMT hard-sphere free energy functionals package.
"""

from src.functionals.base import FMTFunctional
from src.functionals.rosenfeld import RosenfeldFunctional
from src.functionals.white_bear import WhiteBearFunctional, WhiteBearIIFunctional

def functional_factory(name: str) -> FMTFunctional:
    """Factory to obtain a functional instance by name.

    Accepted names (case‑insensitive):
        - "RF"   : Rosenfeld (original) functional
        - "WB"   : White‑Bear functional
        - "WBII" : White‑Bear II functional
    """
    name = name.strip().upper()
    if name == "RF":
        return RosenfeldFunctional()
    if name == "WB":
        return WhiteBearFunctional()
    if name == "WBII":
        return WhiteBearIIFunctional()
    raise ValueError(f"Unknown functional name: {name}")

__all__ = [
    "FMTFunctional",
    "RosenfeldFunctional",
    "WhiteBearFunctional",
    "WhiteBearIIFunctional",
    "functional_factory",
]
