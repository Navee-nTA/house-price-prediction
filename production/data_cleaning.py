"""Processors for the data cleaning step of the worklow.

The processors in this step, apply the various cleaning steps identified
during EDA to create the training datasets.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import mlflow
import mlflow.sklearn

from ta_lib.core.api import (
    custom_train_test_split,
    load_dataset,
    register_processor,
    save_dataset,
    string_cleaning,
)
from scripts import binned_median_income


@register_processor("data-cleaning", "housing")
def clean_housing_table(context, params):
    """Clean the ``housing`` data table.

    The table contains information on the inventory being sold. This
    includes information on inventory id, properties of the item and
    so on.
    """

    input_dataset = "raw/housing"
    output_dataset = "processed/housing"

    # load dataset
    housing_df = load_dataset(context, input_dataset)
    housing_df['income_cat'] = pd.cut(housing_df["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])
    housing_df["rooms_per_household"] = housing_df["total_rooms"]/housing_df["households"]
    housing_df["bedrooms_per_room"] = housing_df["total_bedrooms"]/housing_df["total_rooms"]
    housing_df["population_per_household"]=housing_df["population"]/housing_df["households"]
    housing_df_clean = (
    housing_df
    # while iterating on testing, it's good to copy the dataset(or a subset)
    # as the following steps will mutate the input dataframe. The copy should be
    # removed in the production code to avoid introducing perf. bottlenecks.
    .copy()

    # set dtypes : nothing to do here
    .passthrough()

    .replace({'': np.NaN})
    
    # drop unnecessary cols : nothing to do here
    .drop(columns=['total_bedrooms'])    
    # clean column names (comment out this line while cleaning data above)
    .clean_names(case_type='snake')
    )


    # save the dataset
    save_dataset(context, housing_df_clean, output_dataset)
    # with mlflow.start_run(run_name="Cleaning housing Table"):
    #     mlflow.log_artifact(output_dataset)
    return housing_df_clean

@register_processor("data-cleaning", "train-test")
def create_training_datasets(context, params):
    """Split the ``housing`` table into ``train`` and ``test`` datasets."""

    input_dataset = "processed/housing"
    output_train_features = "train/housing/features"
    output_train_target = "train/housing/target"
    output_test_features = "test/housing/features"
    output_test_target = "test/housing/target"

    # load dataset
    housing_df_processed = load_dataset(context, input_dataset)

    # creating additional features that are not affected by train test split. These are features that are processed globally
    # first time customer(02_data_processing.ipynb)################
    

    # split the data
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=params["test_size"], random_state=context.random_seed
    )
    housing_df_train, housing_df_test = custom_train_test_split(
        housing_df_processed, splitter, by=binned_median_income
    )

    # split train dataset into features and target
    target_col = params["target"]
    train_X, train_y = (
        housing_df_train
        # split the dataset to train and test
        .get_features_targets(target_column_names=target_col)
    )

    # save the train dataset
    save_dataset(context, train_X, output_train_features)
    save_dataset(context, train_y, output_train_target)

    # split test dataset into features and target
    test_X, test_y = (
        housing_df_test
        # split the dataset to train and test
        .get_features_targets(target_column_names=target_col)
    )

    # save the datasets
    save_dataset(context, test_X, output_test_features)
    save_dataset(context, test_y, output_test_target)

    # with mlflow.start_run(run_name="Creating Training Dataset"):
    #     mlflow.log_artifact("~/module5/regression-py/data/" + output_train_features)
    #     mlflow.log_artifact("~/module5/regression-py/data/" + output_train_target)
    #     mlflow.log_artifact("~/module5/regression-py/data/" + output_test_features)
    #     mlflow.log_artifact("~/module5/regression-py/data/" + output_test_target)

    # mlflow.log_artifact()
