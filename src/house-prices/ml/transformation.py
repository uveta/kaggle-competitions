from pandas import DataFrame, Index
from numpy import log1p
from typing import Callable


def unscew_features(
    data: DataFrame,
    index: Index,
    function: Callable[[DataFrame], DataFrame] = lambda x: log1p(x),
    inplace: bool = True,
) -> DataFrame:
    if inplace:
        data[index] = function(data[index])
        return data[index]
    else:
        return function(data[index])
