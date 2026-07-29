"""FMT hard-sphere free energy functionals package.
"""

from src.functionals.base import FMTFunctional
from src.functionals.rosenfeld import RosenfeldFunctional
from src.functionals.tarazona_tensor import WhiteBearTensorFunctional
from src.functionals.white_bear import WhiteBearFunctional, WhiteBearIIFunctional


def functional_factory(name: str) -> FMTFunctional:
    """Factory to obtain a functional instance by name.

    Accepted names (case-insensitive):
        - "RF"        : Rosenfeld (original) functional
        - "WB"        : White-Bear functional
        - "WBII"      : White-Bear II functional
        - "WB-TENSOR" / "WBT" : White-Bear Tensor functional
    """
    name = name.strip().upper()
    if name == "RF":
        return RosenfeldFunctional()
    if name == "WB":
        return WhiteBearFunctional()
    if name == "WBII":
        return WhiteBearIIFunctional()
    if name in ("WB-TENSOR", "WBT", "WB_TENSOR"):
        return WhiteBearTensorFunctional()
    raise ValueError(f"Unknown functional name: {name}")


__all__ = [
    "FMTFunctional",
    "RosenfeldFunctional",
    "WhiteBearFunctional",
    "WhiteBearIIFunctional",
    "WhiteBearTensorFunctional",
    "functional_factory",
]
