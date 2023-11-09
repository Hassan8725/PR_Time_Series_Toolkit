from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from tsai.all import TSStandardScaler, save_object
from tsai.basics import get_forecasting_splits, prepare_forecasting_data


class DataManager:
    """A class to manage data for the toolkit.

    :param data_path: The path to the data file.
    :type data_path: Path
    :param artifacts_path: The path where to
        store the artifacts, defaults to Path("artifacts")
    :type artifacts_path: Path, optional
    """

    def __init__(
        self,
        data_path: Path,
        artifacts_path: Path = Path("artifacts"),
    ) -> None:
        """Constructor Initialization."""
        self.data_path = data_path
        self.artifacts_path = artifacts_path

    def load_data(self) -> pd.DataFrame:
        """Get HD5 Format File.

        :return: data
        :rtype: pd.DataFrame
        """
        data = pd.read_hdf(self.data_path)
        data = data.reset_index()
        return data

    def preprocess_data(
        self,
        data: pd.DataFrame,
        index: str = "DateTime",
        columns: str = "PropertyID",
        values: str = "Value",
    ) -> pd.DataFrame:
        """Reshape and pivot the data.

        :param data: Data to be preprocessed
        :type data: pd.DataFrame
        :param index: The index to use, defaults to "DateTime"
        :type index: str, optional
        :param columns: Column to use as Sensor ID, defaults to "PropertyID"
        :type columns: str, optional
        :param values: Column to use as sensor value, defaults to "Value"
        :type values: str, optional

        :return: Preprocessed data
        :rtype: pd.DataFrame
        """
        data = data.pivot_table(index=index, columns=columns, values=values)
        data.columns = [str(col) for col in data.columns]
        return data

    def interpolate_nulls(
        self,
        data: pd.DataFrame,
        interpolation_method: str = "linear",
        polynomial_order: int = 5,
    ) -> pd.DataFrame:
        """Fill nulls in the data using interpolation.

        :param data: Data with nulls
        :type data: pd.DataFrame
        :param interpolation_method: Interpolation Method.
            All possible Methods can be found at
            `pandas.Dataframe.interpolate() <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html>`_,
            defaults to "linear"
        :type interpolation_method: str, optional
        :param polynomial_order: Order for polynomial interpolation, defaults to 5
        :type polynomial_order: int, optional

        :return: Data with gaps filled
        :rtype: pd.DataFrame
        """
        if interpolation_method in [
            "linear",
            "time",
            "index",
            "values",
            "nearest",
            "zero",
            "slinear",
            "quadratic",
            "cubic",
            "barycentric",
            "krogh",
            "spline",
            "polynomial",
            "from_derivatives",
            "piecewise_polynomial",
            "pchip",
            "akima",
            "cubicspline",
        ]:
            if interpolation_method in ["spline", "polynomial"]:
                data = data.interpolate(
                    method=interpolation_method, order=polynomial_order
                )
                data = data.reset_index()
            else:
                data = data.interpolate(method=interpolation_method)
                data = data.reset_index()
        else:
            raise ValueError("Invalid interpolation method specified.")
        data = data.dropna(axis=0)
        return data

    def split_data(
        self,
        data: pd.DataFrame,
        steps_past: int = 200,
        steps_future: int = 24,
        validation_size: float = 0.1,
        test_size: float = 0.2,
        show_plot: bool = True,
    ) -> tuple:
        """Creates the train/validation/test splits.

        :param data: Data
        :type data: pd.DataFrame
        :param steps_past: The number of steps to look back as history, defaults to 200
        :type steps_past: int, optional
        :param steps_future: The number of steps to look forward, defaults to 24
        :type steps_future: int, optional
        :param validation_size: The size of the validation set, defaults to 0.1
        :type validation_size: float, optional
        :param test_size: The size of the test set, defaults to 0.2
        :type test_size: float, optional
        :param show_plot: Whether to show the split plot, defaults to True
        :type show_plot: bool, optional

        :return: The train/validation/test splits indexes.
        :rtype: tuple
        """
        splits = get_forecasting_splits(
            data,
            fcst_history=steps_past,
            fcst_horizon=steps_future,
            valid_size=validation_size,
            test_size=test_size,
            datetime_col="DateTime",
            show_plot=show_plot,
        )
        return splits

    def standardize_data(
        self,
        data: pd.DataFrame,
        splits: tuple,
        columns: list[str],
        verbose: bool = False,
        artifact_name: str = "standardize_pipe",
    ) -> pd.DataFrame:
        """Standardize the data using a standardization pipeline.

        :param data: Data to be standardized
        :type data: pd.DataFrame
        :param splits: The train/validation/test splits.
        :type splits: tuple
        :param columns: The columns to standardize
        :type columns: list[str]
        :param verbose: Whether to show the pipeline logs, defaults to False
        :type verbose: bool, optional
        :param artifact_name: The name of the artifact to be saved,
            defaults to "standardize_pipe"
        :type artifact_name: str, optional

        :return: Standardized Data
        :rtype: pd.DataFrame
        """
        standardize_pipe = Pipeline(
            [("scaler", TSStandardScaler(columns=columns))], verbose=verbose
        )
        save_object(
            standardize_pipe,
            self.artifacts_path / f"{artifact_name}.pkl",
            verbose=verbose,
        )
        train_split = splits[0]
        standardized_data = standardize_pipe.fit_transform(
            data, scaler__idxs=train_split
        )
        return standardized_data, standardize_pipe

    def get_forecasting_data(
        self, data: pd.DataFrame, steps_past: int = 200, steps_future: int = 24
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gets the forecasting data.

        :param data: Data
        :type data: pd.DataFrame
        :param steps_past: The number of steps to look back, defaults to 200
        :type steps_past: int, optional
        :param steps_future: The number of steps to look forward, defaults to 24
        :type steps_future: int, optional

        :return: The inputs `x` of shape (batch_size, n_vars, seq_len)
            and outputs `y`of shape (batch_size, n_vars, horizon).
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        columns = data.columns[1:]
        x, y = prepare_forecasting_data(
            data,
            fcst_history=steps_past,
            fcst_horizon=steps_future,
            x_vars=columns,
            y_vars=columns,
        )
        return x, y
