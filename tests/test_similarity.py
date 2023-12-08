import pandas as pd
import pytest

from tstoolkit.similarity import TimeSeriesSimilarityChecker


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create a sample DataFrame for testing.

    :return: Sample Dataset
    :rtype: pd.DataFrame
    """
    data = {
        "DateTime": pd.date_range(start="2023-01-01", periods=11, freq="D"),
        "col1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "col2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
    }
    return pd.DataFrame(data)


def test_pearson_correlation(sample_data: pd.DataFrame) -> None:
    """Test Pearson Correlation.

    :param sample_data: sample dataset
    :type sample_data: pd.DataFrame
    """
    similarity_checker = TimeSeriesSimilarityChecker()
    similarity_checker.fit(sample_data)
    correlation = similarity_checker.pearson_correlation("col1", "col2")
    assert round(correlation, 2) == 1.0


def test_cosine_similarity(sample_data: pd.DataFrame) -> None:
    """Test Cosine Similarity.

    :param sample_data: sample dataset
    :type sample_data: pd.DataFrame
    """
    similarity_checker = TimeSeriesSimilarityChecker()
    similarity_checker.fit(sample_data)
    similarity = similarity_checker.cosine_similarity("col1", "col2")
    assert round(similarity, 2) == 1.0


def test_dynamic_time_warping(sample_data: pd.DataFrame) -> None:
    """Test dynamic time wraping.

    :param sample_data: sample dataset
    :type sample_data: pd.DataFrame
    """
    similarity_checker = TimeSeriesSimilarityChecker()
    similarity_checker.fit(sample_data)
    distance = similarity_checker.dynamic_time_warping("col1", "col2")
    assert round(distance, 2) == 41.0
