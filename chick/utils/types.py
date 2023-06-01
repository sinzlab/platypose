from typing import Optional, TypedDict


class Energy(TypedDict):
    train: float
    test: Optional[float]
    val: Optional[float]
