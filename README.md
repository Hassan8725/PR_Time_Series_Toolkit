# Time Series Toolkit

<div align="center">
  <img src="assets/tstoolkit_logo.png" width="1000" height="300" alt="Project Logo">
</div>

![License](https://img.shields.io/badge/License-MIT-blue.svg)

## Table of Contents
- [Introduction](#introduction)
- [Data Preprocessing Module](#data-preprocessing-module)
- [Gap Filling Module](#gap-filling-module)
- [Forecasting Module](#forecasting-module)
- [Anomaly Detection Module](#anomaly-detection-module)
- [Similarity Module](#similarity-module)
- [Setup](#setup)


## Introduction

Welcome to the Time Series Toolkit! This toolkit is designed to assist you in various aspects of time series data analysis and forecasting. Below, we provide detailed explanations for each section of this toolkit.

## Data Preprocessing Module

The Data Preprocessing Module is a fundamental component responsible for preparing your time series data for analysis. It encompasses the following functionalities:

- **Data Preprocessing:** Clean the time series data by removing outliers, missing values, and noise.
- **Interpolation:** Fill in gaps between data points to maintain a consistent time interval.
- **Standardization:** Scale the data to have a mean of 0 and a standard deviation of 1.
- **Splitting Dataset:** Divide the data into training, validation, and test sets for model training and evaluation.
- **Forecasting Dataset Preparation:** Prepare input sequences and target values for forecasting models.

To start using the Data Preprocessing Module, you can import it as follows:
```python
from tstoolkit.data_preprocess import DataManager
```

## Gap Filling Module

The Gap Filling Module addresses the challenge of handling large gaps in time series data. It offers intelligent methods to fill these gaps, ensuring the continuity of the time series.

To start using the Gap Filling Module, you can import it as follows:
```python
from tstoolkit.gap_filling import TimeSeriesGapFiller
```

For detailed tutorial please go [here](tutorials/Gap_Filling.ipynb)

## Forecasting Module

The Forecasting Module is focused on predicting future values of your time series. It leverages two powerful models:

- **Transformer Model:** A deep learning architecture known for handling sequential data and capturing long-range dependencies.
- **Inception Model:** A model capable of identifying complex patterns using convolutional and recurrent layers.

These models enable accurate and efficient time series forecasting.

To start using the Forecasting Module, you can import it as follows:
```python
from tstoolkit.models import TsaiLearnerManager, TsaiModels, SplitType
from tstoolkit.utils import mse, mae, ShowGraph
```

For detailed tutorial please go [here](tutorials/Forecasting_Tutorial.ipynb)

## Anomaly Detection Module

The Anomaly Detection Module is designed to identify anomalies or unusual patterns within your time series data. Anomalies could indicate data collection errors, significant events, or deviations from expected behavior. This module helps you detect and investigate these anomalies for further analysis.

To start using the Anomaly Detection Module, you can import it as follows:
```python
from tstoolkit.anomaly_detection import TimeSeriesAnomalyDetector
```

For detailed tutorial please go [here](tutorials/Anomaly_detection.ipynb)

## Similarity Module

The Similarity Module provides tools for measuring the similarity between two time series. It includes various similarity measures, such as Euclidean distance, Dynamic Time Warping, and Pearson correlation. This functionality allows you to compare and analyze different time series, identifying patterns and correlations.

To start using the Similarity Checking Module, you can import it as follows:
```python
from tstoolkit.similarity import TimeSeriesSimilarityChecker
```

For detailed tutorial please go [here](tutorials/Similarity_checker.ipynb)

In summary, the Time Series Toolkit offers a comprehensive set of tools to handle all aspects of time series data analysis, from preprocessing and gap filling to advanced forecasting, anomaly detection, and similarity measurement.

## Setup

As a dependency manager we make use of [pdm](https://pdm.fming.dev/latest/). Therefore, following are the steps for setting up the project:

```sh
git clone https://github.com/Hassan8725/PR_Ts_Toolkit.git        # clone the repository
cd PR_Ts_Toolkit                                                 # enter the project folder
conda create -n tstoolkit python=3.11                            # create a new conda environment
conda activate tstoolkit                                         # activate the created conda environment
pdm install                                                      # install the dependencies within the environment
```

When you need to add a new dependency, use `pdm add`. For example, to install numpy you run:

```sh
pdm add numpy
```

We also use pre-commit for instantiating various hooks at before commiting/pushing to the repo in order to insure code consistency. Therefore, before commiting/pushing to the remote branch, simply run:

```sh
pre-commit run --all-files
```
