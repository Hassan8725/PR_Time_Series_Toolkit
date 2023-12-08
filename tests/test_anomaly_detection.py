from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tstoolkit.anomaly_detection import TimeSeriesAnomalyDetector
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


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.fixture
def processed_data() -> pd.DataFrame:
    """Fetch and process data for the tests of other modalities.

    :return: processed_data
    :rtype: pd.DataFrame
    """
    data_path = Path("tests/assets/generated_data.hd5")
    data_manager = DataManager(data_path)
    data = data_manager.load_data()
    data = data_manager.preprocess_data(data)
    data = data_manager.interpolate_nulls(data, interpolation_method="linear")
    data = data.set_index("DateTime")

    return data


def test_initialization(processed_data: pd.DataFrame) -> None:
    """Test the initialization of TimeSeriesAnomalyDetector.

    :param processed_data: Processed time series data (pd.DataFrame).
    :type processed_data: pd.DataFrame

    :assertions:
        - The 'data' attribute is set to the processed_data input.
        - The 'anomaly_scores' attribute is an empty list.
        - The 'model' attribute is initially None.
    """
    detector = TimeSeriesAnomalyDetector()

    assert detector.anomaly_scores == []
    assert detector.model is None


def test_fit_with_supported_algorithm(processed_data: pd.DataFrame) -> None:
    """Test the 'fit' method with a supported algorithm.

    :param processed_data: Processed time series data (pd.DataFrame).
    :type processed_data: pd.DataFrame

    :assertions:
        - The 'data' attribute is set to the processed_data input.
    """
    detector = TimeSeriesAnomalyDetector()
    detector.fit(processed_data)
    assert detector.data.equals(processed_data)


def test_predict(processed_data: pd.DataFrame) -> None:
    """Test the 'predict' method.

    :param processed_data: Processed time series data (pd.DataFrame).
    :type processed_data: pd.DataFrame

    :assertions:
        - The length of the returned scores matches the number of columns in the data.
        - The 'model' attribute is not None after fitting with a supported algorithm.

    """
    detector = TimeSeriesAnomalyDetector()
    detector.fit(processed_data)
    scores = detector.predict(algorithm="isolation_forest")
    assert len(scores) == len(processed_data.columns)
    assert detector.model is not None


def test_predict_with_unsupported_algorithm(processed_data: pd.DataFrame) -> None:
    """Test the 'predict' method with an unsupported algorithm.

    :param processed_data: Processed time series data (pd.DataFrame).
    :type processed_data: pd.DataFrame

    :assertions:
        - The 'fit' method raises a ValueError with the expected error message.
    """
    detector = TimeSeriesAnomalyDetector()
    detector.fit(processed_data)

    with pytest.raises(ValueError, match="Unsupported algorithm: some_algorithm"):
        detector.predict(algorithm="some_algorithm")


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
