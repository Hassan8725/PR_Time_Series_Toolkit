import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tstoolkit.data_preprocess import DataManager
from tstoolkit.forecasting_models import SplitType, TsaiLearnerManager, TsaiModels
from tstoolkit.utils import ShowGraph, mae, mse


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
def processed_data() -> list:
    """Fetch and process data for the tests of other modalities.

    :return: List of x, y, splits, pipeline
    :rtype: list
    """
    data_path = Path("tests/assets/generated_data.hd5")
    data_manager = DataManager(data_path)
    data = data_manager.load_data()
    data = data_manager.preprocess_data(data)
    data = data_manager.interpolate_nulls(data, interpolation_method="linear")
    splits = data_manager.split_data(data, show_plot=False)

    data_standardized, standardize_pipe = data_manager.standardize_data(
        data, splits, data.columns[1:], verbose=False
    )

    x, y = data_manager.get_forecasting_data(data_standardized)

    return [x, y, splits, standardize_pipe]


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.fixture
def learner_manager(processed_data: list) -> classmethod:
    """Learner Manager for forecasting Module Tests.

    :param processed_data: Fixture of processed data
    :type processed_data: function
    :return: Learner Manager
    :rtype: type(TsaiLearnerManager)
    """
    arch_patch_tst = TsaiModels.PATCH_TST.value
    x = processed_data[0]
    y = processed_data[1]
    splits = processed_data[2]
    pipe = processed_data[3]

    arch_config_patch_tst = dict(
        n_layers=12,
        n_heads=8,
        d_model=128,
        d_ff=256,
        attn_dropout=0.1,
        dropout=0.1,
        patch_len=24,
        stride=2,
        padding_patch=True,
    )
    run_path = Path(f"./run/{arch_patch_tst}/{uuid.uuid4()}")
    run_path.mkdir(parents=True, exist_ok=True)

    lm = TsaiLearnerManager(
        arch=arch_patch_tst,
        arch_config=arch_config_patch_tst,
        loss_func=mse,
        metrics=[
            mae,
            mse,
        ],
        x=x,
        y=y,
        splits=splits,
        artifacts_path=run_path,
        batch_size=16,
        pipelines=[pipe],
        callbacks=[ShowGraph()],
    )

    return lm


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_split_type_enums() -> None:
    """Test Enums of Split Type."""
    assert SplitType.TRAIN.value == "train"
    assert SplitType.VALID.value == "valid"
    assert SplitType.TEST.value == "test"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_tsaimodels_enum() -> None:
    """Test enums of models."""
    assert TsaiModels.PATCH_TST.value == "PatchTST"
    assert TsaiModels.INCEPTION_TIME_PLUS.value == "InceptionTimePlus"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_mae_with_tensors() -> None:
    """Test MAE."""
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([2.0, 3.0, 4.0])
    result = mae(x, y)
    expected = torch.tensor(1.0)
    torch.testing.assert_close(result, expected)

    x = torch.tensor([0.0, 0.0, 0.0])
    y = torch.tensor([0.0, 0.0, 0.0])
    result = mae(x, y)
    expected = torch.tensor(0.0)
    torch.testing.assert_close(result, expected)

    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.0, 2.0, 3.0])
    result = mae(x, y)
    expected = torch.tensor(0.0)
    torch.testing.assert_close(result, expected)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_mse_with_tensors() -> None:
    """Test MSE."""
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([2.0, 3.0, 4.0])
    result = mse(x, y)
    expected = torch.tensor(1.0)
    torch.testing.assert_close(result, expected)

    x = torch.tensor([0.0, 0.0, 0.0])
    y = torch.tensor([0.0, 0.0, 0.0])
    result = mse(x, y)
    expected = torch.tensor(0.0)
    torch.testing.assert_close(result, expected)

    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.0, 2.0, 3.0])
    result = mse(x, y)
    expected = torch.tensor(0.0)
    torch.testing.assert_close(result, expected)


@pytest.mark.filterwarnings("ignore::DeprecationWarning", "ignore::UserWarning")
def test_find_max_lr(learner_manager: classmethod) -> None:
    """Test find_max_lr function.

    :param learner_manager: Learner Manager Initialiyed Fixture
    :type learner_manager: classmethod
    """
    assert learner_manager.find_max_lr() > 0


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
