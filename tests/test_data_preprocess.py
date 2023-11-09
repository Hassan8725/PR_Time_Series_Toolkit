from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from tstoolkit.data_preprocess import DataManager


def test_create_data() -> None:
    """Create the dataset for tests."""
    # Number of rows in the new dataset
    num_rows = 5000  # Change this to your desired number of rows

    # Start date and time for the dataset
    start_date = datetime(2020, 1, 1, 0, 0, 0)

    # Create an empty DataFrame
    columns = [
        "DateTime",
        "300468",
        "300498",
        "305256",
        "315036",
        "340686",
        "341124",
        "341550",
    ]
    df_struct = pd.DataFrame(columns=columns)

    # Generate data for each column
    for i in range(num_rows):
        row_data = [start_date + timedelta(seconds=i * 30)]
        for _ in range(7):
            row_data.append(
                np.random.uniform(0, 100)
            )  # You can adjust the range as needed
        df_struct.loc[i] = row_data

    # Specify the columns to use as identifiers and variables
    id_vars = ["DateTime"]
    value_vars = ["300468", "300498", "305256", "315036", "340686", "341124", "341550"]

    # Unpivot the DataFrame
    df_unpivoted = df_struct.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="PropertyID",
        value_name="Value",
    )
    df_unpivoted = df_unpivoted[["PropertyID", "DateTime", "Value"]]

    generated_data = df_unpivoted.copy()

    # Define the directory path to save the file
    directory_path = Path("./tests/assets")

    # Create the directory if it doesn't exist
    if not directory_path.exists():
        directory_path.mkdir(parents=True)

    # Define the file path to save the DataFrame
    file_path = directory_path / "generated_data.hd5"

    # Save the DataFrame
    generated_data.to_hdf(file_path, key="data", mode="w")
    pass


@pytest.fixture
def data_manager() -> DataManager:
    """Returns the dataset manager.

    :return: The dataset manager.
    :rtype: DataManager
    """
    return DataManager(data_path="tests/assets/generated_data.hd5")


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_load_data(data_manager: DataManager) -> None:
    """Tests the get_data method.

    :param data_manager: The data manager.
    :type data_manager: DataManager
    """
    data = data_manager.load_data()
    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0
    assert len(data.columns) > 0


@pytest.fixture
def data(data_manager: DataManager) -> pd.DataFrame:
    """Returns the data.

    :param data_manager: The data manager.
    :type data_manager: DataManager

    :return: The data.
    :rtype: pd.DataFrame
    """
    return data_manager.load_data()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_preprocess_data(data_manager: DataManager, data: pd.DataFrame) -> None:
    """Tests the preprocess_data method.

    :param data_manager: The data manager.
    :type data_manager: DataManager
    :param data: The data.
    :type data: pd.DataFrame
    """
    preprocessed_data = data_manager.preprocess_data(data)
    assert isinstance(preprocessed_data, pd.DataFrame)
    assert len(preprocessed_data) > 0
    assert len(preprocessed_data.columns) > 0


@pytest.fixture
def preprocessed_data(data_manager: DataManager, data: pd.DataFrame) -> pd.DataFrame:
    """Returns the pivoted data.

    :param data_manager: The data manager.
    :type data_manager: DataManager
    :param data: The data.
    :type data: pd.DataFrame

    :return: The preprocessed_data.
    :rtype: pd.DataFrame
    """
    return data_manager.preprocess_data(data=data)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_interpolate_nulls(
    data_manager: DataManager, preprocessed_data: pd.DataFrame
) -> None:
    """Tests the interpolate nulls method.

    :param data_manager: The data manager.
    :type data_manager: DataManager
    :param preprocessed_data: The preprocessed_data.
    :type preprocessed_data: pd.DataFrame
    """
    result = data_manager.interpolate_nulls(
        preprocessed_data, interpolation_method="linear"
    )
    assert result.isna().sum().sum() == 0


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_interpolate_nulls_invalid_method(
    data_manager: DataManager, preprocessed_data: pd.DataFrame
) -> None:
    """Tests the interpolate nulls method with an invalid method.

    :param data_manager: The data manager.
    :type data_manager: DataManager
    :param preprocessed_data: The preprocessed_data.
    :type preprocessed_data: pd.DataFrame
    """
    with pytest.raises(ValueError):
        data_manager.interpolate_nulls(
            preprocessed_data, interpolation_method="invalid_method"
        )


@pytest.fixture
def interpolated_data(
    data_manager: DataManager, preprocessed_data: pd.DataFrame
) -> pd.DataFrame:
    """Returns the interpolated data.

    :param data_manager: The data manager.
    :type data_manager: DataManager
    :param preprocessed_data: The preprocessed_data.
    :type preprocessed_data: pd.DataFrame

    :return: The interpolated data.
    :rtype: pd.DataFrame
    """
    return data_manager.interpolate_nulls(
        preprocessed_data, interpolation_method="linear"
    )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_create_splits(
    data_manager: DataManager, interpolated_data: pd.DataFrame
) -> None:
    """Tests the create_splits method.

    :param data_manager: The data manager.
    :type data_manager: DataManager
    :param interpolated_data: The interpolated_data.
    :type interpolated_data: pd.DataFrame
    """
    splits = data_manager.split_data(interpolated_data, show_plot=False)
    assert isinstance(splits, tuple)
    assert len(splits[0]) > 0
    assert len(splits[1]) > 0
    assert len(splits[2]) > 0


@pytest.fixture
def splits(data_manager: DataManager, interpolated_data: pd.DataFrame) -> tuple:
    """Returns the splits.

    :param data_manager: The data manager.
    :type data_manager: DataManager
    :param interpolated_data: The interpolated_data.
    :type interpolated_data: pd.DataFrame

    :return: The splits.
    :rtype: tuple
    """
    return data_manager.split_data(data=interpolated_data, show_plot=False)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_standardize_data(
    data_manager: DataManager,
    interpolated_data: pd.DataFrame,
    splits: tuple,
) -> None:
    """Tests the get_standardization_pipeline method.

    :param data_manager: The data manager.
    :type data_manager: DataManager
    :param interpolated_data: The interpolated_data.
    :type interpolated_data: pd.DataFrame
    :param splits: The splits.
    :type splits: tuple
    """
    standardized_data, standardization_pipe = data_manager.standardize_data(
        interpolated_data, splits, interpolated_data.columns[1:]
    )
    assert isinstance(standardization_pipe, Pipeline)
    assert isinstance(standardized_data, pd.DataFrame)
    assert len(standardized_data) > 0
    assert len(standardized_data.columns) > 0


@pytest.fixture
def get_standardized_data_and_pipeline(
    data_manager: DataManager,
    interpolated_data: pd.DataFrame,
    splits: tuple,
) -> list:
    """Returns the standardized data and pipeline.

    :param data_manager: The data manager.
    :type data_manager: DataManager
    :param interpolated_data: The interpolated_data.
    :type interpolated_data: pd.DataFrame
    :param splits: The splits.
    :type splits: tuple

    :return: The standardized data and pipeline.
    :rtype: list
    """
    standardized_data, standardization_pipe = data_manager.standardize_data(
        interpolated_data, splits, interpolated_data.columns[1:]
    )

    return [standardized_data, standardization_pipe]


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_get_forecasting_data(
    data_manager: DataManager,
    get_standardized_data_and_pipeline: list,
) -> None:
    """Tests the get_forecasting_data method.

    :param data_manager: The data manager.
    :type data_manager: DataManager
    :param get_standardized_data_and_pipeline: The standardized data and pipeline.
    :type get_standardized_data_and_pipeline: list
    """
    standardized_data = get_standardized_data_and_pipeline[0]
    x, y = data_manager.get_forecasting_data(standardized_data)
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(x) == len(y)


@pytest.fixture
def forecasting_data(
    data_manager: DataManager, standardized_data: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the forecasting data.

    :param data_manager: The data manager.
    :type data_manager: DataManager
    :param standardized_data: The standardized data.
    :type standardized_data: pd.DataFrame

    :return: The inputs `x` and outputs `y`.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    return data_manager.get_forecasting_data(standardized_data)


def test_remove_data() -> None:
    """Remove the created dataset for tests."""
    # Define the directory path to save the file
    directory_path = Path("./tests/assets")
    file_path = directory_path / "generated_data.hd5"

    # Remove the file if it exists
    if file_path.exists():
        file_path.unlink()

    # Remove the directory if it exists
    if directory_path.exists():
        directory_path.rmdir()
    pass
