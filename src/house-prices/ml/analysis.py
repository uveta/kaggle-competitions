from pandas import DataFrame, Index, Series
from scipy.stats import skew


def get_numeric_features(data: DataFrame) -> Series:
    return data.dtypes[data.dtypes != "object"].index


def get_feature_scews(data: DataFrame) -> Series:
    return data[get_numeric_features(data)].apply(lambda feat: skew(feat), axis="index")


def get_scewed_features_index(data: DataFrame, limit: float = 0.75) -> Index:
    scewed_features = get_feature_scews(data)
    scewed_features = scewed_features[scewed_features > limit]
    return scewed_features.index
