import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


class TimeSeriesAnomalyDetector:
    """A class for detecting anomalies within a timeseries."""

    def __init__(
        self,
    ) -> None:
        """Constructor Initialization."""
        self.anomaly_scores = []
        self.model = None

    def fit(
        self,
        data: pd.DataFrame,
    ) -> None:
        """Fit an anomaly detection model to the time series data.

        :param data: Timeseries DataFrame.
        :type data: pd.DataFrame
        """
        self.data = data
        self.time_series_columns = data.columns

    def predict(
        self,
        algorithm: str = "isolation_forest",
    ) -> list:
        """Predict anomalies in the time series data.

        :param algorithm: The algorithm to use
            for anomaly detection. Options available are
            isolation_forest, local_outlier_factor
            and one_class_svm, defaults to "isolation_forest"
        :type algorithm: str, optional

        :raises ValueError: Unsupported algorithm

        :return: List of anomaly scores
        :rtype: list
        """
        if algorithm == "isolation_forest":
            self.model = IsolationForest(contamination="auto", random_state=42)
        elif algorithm == "local_outlier_factor":
            self.model = LocalOutlierFactor(contamination="auto")
        elif algorithm == "one_class_svm":
            self.model = OneClassSVM()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        for col in self.time_series_columns:
            anomalies = self.model.fit_predict(self.data[col].to_numpy().reshape(-1, 1))
            self.anomaly_scores.append(anomalies)

        return self.anomaly_scores

    def plot_anomalies(self) -> None:
        """Plot the time series data with detected anomalies highlighted."""
        for i, col in enumerate(self.time_series_columns):
            plt.figure(figsize=(20, 10))
            normal_color = "green"
            anomaly_color = "red"
            normal_mask = self.anomaly_scores[i] == 1
            anomaly_mask = self.anomaly_scores[i] == -1
            plt.plot(self.data.index, self.data[col], label=col, color=normal_color)
            plt.scatter(
                self.data.index[normal_mask],
                self.data[col][normal_mask],
                c=normal_color,
                label="Normal",
                alpha=0.5,
            )
            plt.scatter(
                self.data.index[anomaly_mask],
                self.data[col][anomaly_mask],
                c=anomaly_color,
                label="Anomaly",
                alpha=0.7,
            )
            plt.ylabel("Value")
            plt.title(f"Anomaly Detection in {col}")
            plt.legend()
            plt.tight_layout()
            plt.show()
