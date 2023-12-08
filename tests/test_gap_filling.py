import pandas as pd
import pytest

from tstoolkit.gap_filling import TimeSeriesGapFiller


@pytest.fixture
def gap_filler() -> tuple:
    """Gap Filler Fixture for tests.

    :return: Tuple of Gap Filler Instance, First TS and Second TS
    :rtype: tuple
    """
    first_ts = pd.Series(
        [1, 2, 3, 4], index=pd.date_range(start="2023-01-01", periods=4, freq="D")
    )
    second_ts = pd.Series(
        [10, 20, 30, 40], index=pd.date_range(start="2023-01-05", periods=4, freq="D")
    )
    return (TimeSeriesGapFiller(seasonal_periods=2), first_ts, second_ts)


def test_constructor_default_seasonal_periods() -> None:
    """Test Constructor default values.

    :assertions:
        - The 'seasonal_periods' attribute is set to the 60.
    """
    gap_filler = TimeSeriesGapFiller()
    assert gap_filler.seasonal_periods == 60


def test_constructor_custom_seasonal_periods() -> None:
    """Test Constructor custom seasonal values.

    :assertions:
        - Custom 'seasonal_periods' attribute is set to the the input provided.
    """
    gap_filler = TimeSeriesGapFiller(seasonal_periods=30)
    assert gap_filler.seasonal_periods == 30


def test_fit(gap_filler: tuple) -> None:
    """Test Fit Function.

    :param gap_filler: Tuple with instance and parts of timeseries
    :type gap_filler: tuple

    :assertions:
        - Check if'filled_series' instance is a series.
    """
    gap_filler_obj = gap_filler[0]
    first_ts = gap_filler[1]
    second_ts = gap_filler[2]
    gap_filler_obj.fit(first_ts, second_ts)

    assert len(gap_filler_obj.firsttsr.index) == len(gap_filler_obj.secondtsr.index)


def test_predict(gap_filler: tuple) -> None:
    """Test Predict Function.

    :param gap_filler: Tuple with instance and parts of timeseries
    :type gap_filler: tuple

    :assertions:
        - Check if'filled_series' instance is a series.
    """
    gap_filler_obj = gap_filler[0]
    first_ts = gap_filler[1]
    second_ts = gap_filler[2]
    gap_filler_obj.fit(first_ts, second_ts)
    filled_series = gap_filler_obj.predict()
    assert isinstance(filled_series, pd.Series)


def test_merge_ts(gap_filler: tuple) -> None:
    """Test merge timeseries.

    :param gap_filler: Tuple with instance and parts of timeseries
    :type gap_filler: tuple

    :assertions:
        - Check if'merged_series' instance is a series.
    """
    gap_filler_obj = gap_filler[0]
    first_ts = gap_filler[1]
    second_ts = gap_filler[2]
    gap_filler_obj.fit(first_ts, second_ts)
    gap_filler_obj.predict()
    merged_series = gap_filler_obj.merge_ts()
    assert isinstance(merged_series, pd.Series)
