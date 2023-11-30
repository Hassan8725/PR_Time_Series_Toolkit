from collections.abc import Callable, Iterable
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastai.callback.core import Callback
from sklearn.pipeline import Pipeline
from tsai.all import TSForecaster, load_learner, plot_forecast


class LearnerManager:
    """A class that manages the learners."""

    def __init__(self) -> None:
        """Initializes the learner manager."""
        pass


class SplitType(Enum):
    """An enum that represents the split type."""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class TsaiModels(Enum):
    """An enum that represents the type of Tsai model."""

    PATCH_TST = "PatchTST"
    INCEPTION_TIME_PLUS = "InceptionTimePlus"


class TsaiLearnerManager(LearnerManager):
    """A class that manages the Tsai learners.

    :param arch: The architecture to use.
    :type arch: str
    :param arch_config: The architecture configuration.
    :type arch_config: dict
    :param loss_func: The loss function for training.
    :type loss_func: Callable
    :param metrics: Evaluation metrics.
    :type metrics: list[Callable]
    :param x: The inputs x.
    :type x: np.ndarray
    :param y: The inputs y.
    :type y: np.ndarray
    :param splits: The train/val/test splits.
    :type splits: tuple
    :param artifacts_path: The path where to store
        the artifacts (checkpoints).
    :type artifacts_path: Path
    :param batch_size: The batch size.
    :type batch_size: int
    :param pipelines: The data-preprocessing pipelines.
    :type pipelines: Iterable[Pipeline], optional
    :param callbacks: The callbacks used during training.
    :type callbacks: Iterable[Callback], optional
    """

    def __init__(
        self,
        arch: str,
        arch_config: dict,
        loss_func: Callable,
        metrics: list[Callable],
        x: np.ndarray,
        y: np.ndarray,
        splits: tuple,
        artifacts_path: Path,
        batch_size: int,
        pipelines: Iterable[Pipeline] = ((),),
        callbacks: Iterable[Callback] = ((),),
        **kwargs,
    ) -> None:
        """Constructor Initialization."""
        super().__init__()

        self.learner: TSForecaster = TSForecaster(
            X=x,
            y=y,
            splits=splits,
            path=artifacts_path,
            batch_size=batch_size,
            pipelines=pipelines,
            arch=arch,
            arch_config=arch_config,
            loss_func=loss_func,
            metrics=metrics,
            cbs=callbacks,
            **kwargs,
        )

        self.metrics = metrics

    def get_summary(self) -> str:
        """Returns the summary of the learner.

        :return: Summary
        :rtype: str
        """
        return self.learner.summary()

    def find_max_lr(self) -> float:
        """Find the maximum learning rate.

        :return: Max Learning Rate
        :rtype: float
        """
        return self.learner.lr_find().valley

    def fit(self, epochs: int, lr_max: float | None = None) -> None:
        """Fits the model.

        :param epochs: Number of Epochs
        :type epochs: int
        :param lr_max: Learning Rate, defaults to None
        :type lr_max: float | None, optional
        """
        if not lr_max:
            lr_max = self.find_max_lr()

        self.learner.fit_one_cycle(epochs, lr_max=lr_max)

    def export_model(self, artifact_name: str) -> None:
        """Exports the model.

        :param artifact_name: Name of the artifact to be saved.
        :type artifact_name: str
        """
        self.learner.export(artifact_name)

    def load_model(self, artifact_name: str) -> None:
        """Loads the model.

        :param artifact_name: Name of the artifact to be loaded.
        :type artifact_name: str
        """
        self.learner = load_learner(artifact_name)

    def evaluate(
        self, x: np.ndarray, y_true: np.ndarray, split_type: SplitType = SplitType.VALID
    ) -> pd.DataFrame:
        """Evaluates the validation set.

        :param x: The inputs of shape (batch_size, n_vars, seq_len)
        :type x: np.ndarray
        :param y_true: The targets of shape (batch_size, n_vars, horizon)
        :type y_true: np.ndarray
        :param split_type: The split type., defaults to SplitType.VALID
        :type split_type: SplitType, optional

        :return: Evaluation Metrics Value
        :rtype: pd.DataFrame
        """
        y_pred, *_ = self.learner.get_X_preds(x)
        y_true = torch.from_numpy(y_true)

        metric_names = [metric.__name__ for metric in self.metrics]

        results_df = pd.DataFrame(columns=metric_names)
        for metric_name, metric in zip(metric_names, self.metrics, strict=True):
            results_df.loc[split_type.value, metric_name] = metric(
                y_true, y_pred
            ).item()

        return results_df

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts the targets.

        :param x: The inputs of shape (batch_size, n_vars, seq_len)
        :type x: np.ndarray

        :return: The predictions of shape (batch_size, n_vars, horizon)
        :rtype: np.ndarray
        """
        y_pred, *_ = self.learner.get_X_preds(x)
        return y_pred.numpy()

    def get_forecast_plot(
        self,
        x: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sel_vars: bool = True,
        idx: int = 0,
    ) -> None:
        """Gets the forecast plot.

        :param x:  The inputs of shape (batch_size, n_vars, seq_len)
        :type x: np.ndarray
        :param y_true: The targets of shape (batch_size, n_vars, horizon)
        :type y_true: np.ndarray
        :param y_pred: The predictions of shape (batch_size, n_vars, horizon)
        :type y_pred: np.ndarray
        :param sel_vars: Whether to plot the variable separately., defaults to True
        :type sel_vars: bool, optional
        :param idx: The index of the sample in the batch to plot., defaults to 0
        :type idx: int, optional
        """
        plot_forecast(x, y_true, y_pred, sel_vars=sel_vars, idx=idx)
