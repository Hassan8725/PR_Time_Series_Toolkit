import matplotlib.pyplot as plt
import pandas as pd
from fastdtw import fastdtw
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity


class TimeSeriesSimilarityChecker:
    """A class for performing similarity checks on time series data."""

    def __init__(self) -> None:
        """Constructor Initialization."""

    def fit(self, data: pd.DataFrame) -> None:
        """Fit Similarity Checker.

        :param data: DataFrame containing time series data with DateTime index.
        :type data: pd.DataFrame
        """
        self.data = data

    def pearson_correlation(self, col1: str, col2: str) -> float:
        """Calculate Pearson correlation coefficient between two columns.

        :param col1: Name of the first column.
        :type col1: str
        :param col2: Name of the second column.
        :type col2: str

        :return: Pearson correlation coefficient.
        :rtype: float
        """
        correlation, _ = pearsonr(self.data[col1], self.data[col2])
        return correlation

    def cosine_similarity(self, col1: str, col2: str) -> float:
        """Calculate cosine similarity between two columns.

        :param col1: Name of the first column.
        :type col1: str
        :param col2: Name of the second column.
        :type col2: str

        :return: Cosine similarity.
        :rtype: float
        """
        similarity = cosine_similarity(
            self.data[col1].to_numpy().reshape(1, -1),
            self.data[col2].to_numpy().reshape(1, -1),
        )
        return similarity[0, 0]

    def dynamic_time_warping(self, col1: str, col2: str) -> float:
        """Calculate Dynamic Time Warping (DTW) distance between two columns.

        :param col1: Name of the first column.
        :type col1: str
        :param col2: Name of the second column.
        :type col2: str

        :return: DTW distance.
        :rtype: float
        """
        distance, _ = fastdtw(self.data[col1], self.data[col2])
        return distance

    def plot_signals(self, col1: str, col2: str) -> None:
        """Plot two columns from the time series data.

        :param col1: Name of the first column.
        :type col1: str
        :param col2: Name of the second column.
        :type col2: str
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.data.index, self.data[col1], label=col1)
        plt.plot(self.data.index, self.data[col2], label=col2)
        plt.xlabel("DateTime")
        plt.ylabel("Value")
        plt.legend()
        plt.title(f"Plot of {col1} and {col2}")
        plt.show()
