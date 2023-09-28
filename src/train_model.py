"""This is script trains the model using the data that is processed in process.py.

It has the following functions:
    * main_flow - function from which the flow starts
    * configure_mlflow - configures MLflow on the EC2 instance
    * load_data - loads the processed data
    * train_model - trains the model
    * evaluate_model - returns the scores of the model
    * make_prefect_report - makes a report of the results for Prefect

Author: Emile Lampe
"""

import pickle
from datetime import datetime

import boto3
import mlflow
import pandas as pd
import statsmodels.api as sm
from hydra import compose, initialize
from omegaconf import DictConfig
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@flow
def main_flow():
    """Starts the main flow.

    Args:
        None
    Returns:
        None
    """

    logger = get_run_logger()

    config = retrieve_config()

    logger.info("Config retrieved")

    model_name = config.model.name

    logger.info(f"Experiment: {config.mlflow.experiment}")
    logger.info(f"Train modeling using {config.data.processed}")
    logger.info(f"Model used: {model_name}")

    # Configure MLflow on the EC2 instance
    configure_mlflow(
        aws_profile=config.aws.profile,
        ec2_tags=config.aws.ec2_tags,
        mlflow_port=config.mlflow.port,
        mlflow_experiment=config.mlflow.experiment,
    )

    logger.info("MLflow configured")

    X_train, y_train, X_test, y_test = load_data(path=config.data.processed)

    logger.info("Data loaded")

    with mlflow.start_run():
        ols = train_model(X_train=X_train, y_train=y_train)

        logger.info("Model finished training")

        predictions = ols.predict(X_test)
        mae, mse, r2 = evaluate_model(y_test=y_test, predictions=predictions)

        log_mlflow_metrics(config.metric_names, mae, mse, r2)

        logger.info("Metrics logged on MLflow")

        log_mlflow_model(model=ols, model_name=model_name)

        logger.info("Model logged on MLflow and uploaded to S3")

    make_prefect_report(
        experiment_name=config.mlflow.experiment,
        model_name=model_name,
        data_path=config.data.processed,
        mae=mae,
        mse=mse,
        r2=r2,
    )

    logger.info("Report created")


@task(retries=3, retry_delay_seconds=2)
def retrieve_config() -> DictConfig:
    """Retrieves the config from the config file.

    Args:
        None
    Returns:
        config: The config.
    """

    with initialize(version_base=None, config_path="../config"):
        config = compose(config_name="main")

    return config


@task(retries=3, retry_delay_seconds=2)
def configure_mlflow(
    aws_profile: str, ec2_tags: DictConfig, mlflow_port: int, mlflow_experiment: str
) -> None:
    """Configures MLflow on the EC2 instance.

    Args:
        aws_profile: The AWS profile to get access to the EC2 instance.
        ec2_tags: The EC2 tags to find the EC2 instance.
        mlflow_port: The port on which MLflow is running.
        mlflow_experiment: The name of the MLflow experiment.
    Returns:
        None
    """

    # Create an EC2 client
    ec2_client = boto3.Session(profile_name=aws_profile).client("ec2")

    # Get the response of the EC2 client based on its tags
    filters = []

    for key, value in ec2_tags.items():
        filters.append({"Name": f"tag:{key}", "Values": [value]})

    response = ec2_client.describe_instances(Filters=filters)

    # Get the public IP address of the currenb  t EC2 instance
    public_ip = response["Reservations"][0]["Instances"][0]["PublicIpAddress"]

    remote_tracking_uri = f"http://{public_ip}:{mlflow_port}/"

    mlflow.set_tracking_uri(remote_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)


@task(retries=3, retry_delay_seconds=2)
def load_data(path: str) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    """Loads the processed data from the pickle file.

    Args:
        path: Path to the data.
    Returns:
        X_train: Training data.
        y_train: Training target.
        X_test: Testing data.
        y_test: Testing target.
    """

    # Open X_y_data.pickle
    with open(path, "rb") as f:
        X_train, y_train, X_test, y_test = pickle.load(f)

    return X_train, y_train, X_test, y_test


@task
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> any:
    """Trains the model.

    Args:
        X_train: Training data.
        y_train: Training target.
    Returns:
        ols: Trained model.
    """
    ols = sm.OLS(y_train, X_train).fit()

    return ols


@task
def evaluate_model(y_test: pd.Series, predictions: pd.Series) -> (float, float, float):
    """Evaluates the model with the chosen metrics.

    Args:
        y_test: Real values of the test set.
        predictions: Predictions for the test set.
    Returns:
        mae: Mean absolute error.
        mse: Mean squared error.
        r2: R-squared.
    """

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return mae, mse, r2


@task
def log_mlflow_metrics(
    metric_names: DictConfig, mae: float, mse: float, r2: float
) -> None:
    mlflow.log_metric(metric_names.mean_absolute_error, mae)
    mlflow.log_metric(metric_names.mean_squared_error, mse)
    mlflow.log_metric(metric_names.r2_score, r2)


@task
def log_mlflow_model(model: any, model_name: str) -> None:
    """Logs the model to MLflow and stores it in the S3 bucket.

    Args:
        ols: Trained model.
        model_name: Name of the model.
    Returns:
        None
    """

    mlflow.statsmodels.log_model(
        statsmodels_model=model,
        artifact_path=model_name,
        registered_model_name=model_name,
    )


@task
def make_prefect_report(
    experiment_name: str,
    model_name: str,
    data_path: str,
    mae: float,
    mse: float,
    r2: float,
) -> None:
    """Makes a report of the results for Prefect.

    Args:
        experiment_name: Name of the MLflow experiment.
        model_name: Name of the model.
        data_path: Path to the data.
        mae: Mean absolute error.
        mse: Mean squared error.
        r2: R-squared.
    Returns:
        None
    """

    markdown_report = f"""# Score overview

Experiment: {experiment_name}
Model: {model_name}
Data: {data_path}

The scores for the model are:
- Mean absolute error: {round(mae, 3)}
- Mean squared error: {round(mse, 3)}
- R-squared: {round(r2, 3)}

Time of execution: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
"""

    create_markdown_artifact(
        key="result-report",
        markdown=markdown_report,
        description="Results of the model",
    )


if __name__ == "__main__":
    main_flow.serve(name="model-deployment")
    # main_flow()
