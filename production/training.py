"""Processors for the model training step of the worklow."""

import logging
import os.path as op

from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.preprocessing import FunctionTransformer

from ta_lib.core.api import (
    get_dataframe,
    get_feature_names_from_column_transformer,
    get_package_path,
    load_dataset,
    load_pipeline,
    register_processor,
    save_pipeline,
    DEFAULT_ARTIFACTS_PATH,
)
from ta_lib.regression.api import SKLStatsmodelOLS

logger = logging.getLogger(__name__)


import mlflow.pyfunc
import pickle
import os


class StatsModelsWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)


def save_model_to_mlflow(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


# @register_processor("model-gen", "train-model")
# def train_model(context, params):
#     """Train a regression model."""
#     artifacts_folder = DEFAULT_ARTIFACTS_PATH

#     input_features_ds = "train/housing/features"
#     input_target_ds = "train/housing/target"

#     # load training datasets
#     train_X = load_dataset(context, input_features_ds)
#     train_y = load_dataset(context, input_target_ds)

#     # load pre-trained feature pipelines and other artifacts
#     curated_columns = load_pipeline(op.join(artifacts_folder, "curated_columns.joblib"))
#     features_transformer = load_pipeline(op.join(artifacts_folder, "features.joblib"))

#     # sample data if needed. Useful for debugging/profiling purposes.
#     sample_frac = params.get("sampling_fraction", None)
#     if sample_frac is not None:
#         logger.warn(f"The data has been sample by fraction: {sample_frac}")
#         sample_X = train_X.sample(frac=sample_frac, random_state=context.random_seed)
#     else:
#         sample_X = train_X
#     sample_y = train_y.loc[sample_X.index]

#     # transform the training data
#     train_X = get_dataframe(
#         features_transformer.fit_transform(train_X, train_y),
#         get_feature_names_from_column_transformer(features_transformer),
#     )
#     train_X = train_X[curated_columns]

#     # create training pipeline
#     reg_ppln_ols = Pipeline([("estimator", SKLStatsmodelOLS())])

#     # fit the training pipeline
#     reg_ppln_ols.fit(train_X, train_y.values.ravel())

#     # save fitted training pipeline
#     save_pipeline(
#         reg_ppln_ols, op.abspath(op.join(artifacts_folder, "train_pipeline.joblib"))
#     )

#     # Assuming `model` is your trained statsmodels model
#     model_path = "linear_regression.pkl"
#     save_model_to_mlflow(reg_ppln_ols, model_path)

#     with mlflow.start_run(run_name="Training") as run:
#         # Log the model file as an artifact
#         mlflow.log_artifact(
#             op.abspath(op.join(artifacts_folder, "train_pipeline.joblib"))
#         )
#         mlflow.log_artifact(model_path)

@register_processor("model-gen", "train-model")
def train_model(context, params):
    """Train a regression model."""
    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    input_features_ds = "train/housing/features"
    input_target_ds = "train/housing/target"

    # load training datasets
    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)

    # load pre-trained feature pipelines and other artifacts
    curated_columns = load_pipeline(op.join(artifacts_folder, "curated_columns.joblib"))
    features_transformer = load_pipeline(op.join(artifacts_folder, "features.joblib"))

    # sample data if needed. Useful for debugging/profiling purposes.
    sample_frac = params.get("sampling_fraction", None)
    if sample_frac is not None:
        logger.warn(f"The data has been sample by fraction: {sample_frac}")
        sample_X = train_X.sample(frac=sample_frac, random_state=context.random_seed)
    else:
        sample_X = train_X
    sample_y = train_y.loc[sample_X.index]

    # transform the training data
    train_X = get_dataframe(
        features_transformer.fit_transform(train_X, train_y),
        get_feature_names_from_column_transformer(features_transformer),
    )
    train_X = train_X[curated_columns]

    # create training pipeline
    estimator = XGBRegressor(gamma = params.get('gamma'),
                            learning_rate = params.get('learning_rate'),
                            max_depth = params.get('max_depth'),
                            min_child_weight = params.get('min_child_weight'),
                            n_estimators= params.get('n_estimators') )
    
    xgb_training_pipe2 = Pipeline([
    ('XGBoost', XGBRegressor())
    ])
   

    # fit the training pipeline
    xgb_training_pipe2.fit(train_X, train_y.values.ravel())

    # save fitted training pipeline
    save_pipeline(
        xgb_training_pipe2, op.abspath(op.join(artifacts_folder, "train_pipeline.joblib"))
    )

    # Assuming `model` is your trained statsmodels model
    model_path = "xgb.pkl"
    save_model_to_mlflow(xgb_training_pipe2, model_path)
    
    with mlflow.start_run(run_name="Training") as run:
        # Log the model file as an artifact
        mlflow.log_artifact(
            op.abspath(op.join(artifacts_folder, "train_pipeline.joblib"))
        )
        mlflow.log_artifact(model_path)
