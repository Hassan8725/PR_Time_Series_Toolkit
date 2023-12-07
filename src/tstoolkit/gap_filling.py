from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class TimeSeriesGapFiller:
    """A class for Filling Gaps in Time Series data.

    :param seasonal_periods: The number of periods in a
        complete seasonal cycle, e.g. 4 for quarterly data or 7
        for daily data with a weekly cycle, defaults to 60
    :type seasonal_periods: int, optional
    """

    def __init__(self, seasonal_periods: int = 60) -> None:
        """Constructor Initialization."""
        self.seasonal_periods = seasonal_periods

    def fit(self, first_ts: pd.Series, second_ts: pd.Series) -> None:
        """Fitting Gap Filling Method.

        :param first_ts: Part of Time series before the gap.
        :type first_ts: pd.Series
        :param second_ts: Part of time series after the gap.
        :type second_ts: pd.Series
        """
        self.first_ts = first_ts
        self.second_ts = second_ts

        # PREPARATION
        self.one = timedelta(minutes=1)
        self.secondtsr = self.second_ts[::-1].copy()
        self.firsttsr = self.first_ts[::-1].copy()
        self.indexr = pd.date_range(
            start=self.first_ts.index[0], end=self.second_ts.index[-1], freq="min"
        )
        self.firsttsr.index = self.indexr[-len(self.firsttsr) :]
        self.secondtsr.index = self.indexr[: len(self.secondtsr)]
        # FORWARD
        self.es1 = ExponentialSmoothing(
            self.first_ts, seasonal_periods=self.seasonal_periods, seasonal="add"
        ).fit()

        # BACKWARD
        self.es2 = ExponentialSmoothing(
            self.secondtsr, seasonal_periods=self.seasonal_periods, seasonal="add"
        ).fit()

    def predict(self) -> pd.Series:
        """Getting the filled time series.

        :return: Filled Time Series
        :rtype: pd.Series
        """
        # FORWARD
        forward_prediction = self.es1.predict(
            start=self.first_ts.index[-1] + self.one,
            end=self.second_ts.index[0] - self.one,
        )

        # BACKWARD
        backward_prediction = self.es2.predict(
            start=self.secondtsr.index[-1] + self.one,
            end=self.firsttsr.index[0] - self.one,
        )

        # WEIGHTED_AVERAGE
        length = len(forward_prediction)
        filled_series = pd.Series(
            [
                (backward_prediction[i] * i + forward_prediction[i] * (length - i))
                / length
                for i in range(length)
            ],
            index=forward_prediction.index.copy(),
        )

        self.filled_series = filled_series

        return self.filled_series

    def plot_ts(self) -> None:
        """Plot all three parts of time series combined."""
        plt.figure(figsize=(12, 6))  # Adjust the figure size as needed

        plt.plot(self.first_ts, color="C1", label="First TS")
        plt.plot(self.filled_series, color="C4", label="Filled Gap")
        plt.plot(self.second_ts, color="C2", label="Second TS")

        plt.ylabel("Value")
        plt.title("Filled Series")
        plt.legend()
        plt.xticks(rotation=20)

        plt.show()

    def merge_ts(self) -> pd.Series:
        """Returning a merged time series of all three parts.

        :return: Merged Time series
        :rtype: pd.Series
        """
        return pd.concat([self.first_ts, self.filled_series, self.second_ts])
