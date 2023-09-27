"""
This is script trains the model using the data that is processed in process.py.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    dra to access the parameters in under the directory config.

Author: Emile Lampe
"""

import pickle
from datetime import datetime

import boto3
import hydra
import mlflow
import pandas as pd
import statsmodels.api as sm
from omegaconf import DictConfig
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@hydra.main(config_path="../config", config_name="main", version_base=None)
@flow(name="Train Model", retries=3, retry_delay_seconds=2, log_prints=True)
def main_flow(config: DictConfig):
    """Function from which the flow starts.

    Args:
        None
    Returns:
        None
    """

    # Configure MLflow on the EC2 instance
    configure_mlflow(
        aws_profile=config.aws.profile,
        ec2_tags=config.aws.ec2_tags,
        mlflow_port=config.mlflow.port,
        mlflow_experiment=config.mlflow.experiment,
    )

    print("MLflow configured")

    model_name = config.model.name

    print(f"Train modeling using {config.data.processed}")
    print(f"Model used: {model_name}")
    print(f"Save the output to {config.data.final}")

    X_train, y_train, X_test, y_test = load_data(path=config.data.processed)

    print("Data loaded")

    with mlflow.start_run():
        print("Model training started")

        ols = train_model(X_train=X_train, y_train=y_train)

        print("Model trained")

        predictions = ols.predict(X_test)
        mae, mse, r2 = evaluate_model(y_test=y_test, predictions=predictions)

        # log metrics
        mlflow.log_metric(config.metric_names.mean_absolute_error, mae)
        mlflow.log_metric(config.metric_names.mean_squared_error, mse)
        mlflow.log_metric(config.metric_names.r2_score, r2)

        mlflow.statsmodels.log_model(
            statsmodels_model=ols,
            artifact_path=model_name,
            registered_model_name=model_name,
        )

        print("Model logged")

    make_prefect_report(
        experiment_name=config.mlflow.experiment,
        model_name=model_name,
        data_path=config.data.processed,
        mae=mae,
        mse=mse,
        r2=r2,
    )


@task(name="Configure MLflow", retries=3, retry_delay_seconds=2)
def configure_mlflow(
    aws_profile: str, ec2_tags: DictConfig, mlflow_port: int, mlflow_experiment: str
) -> None:
    """Function to configure MLflow on the EC2 instance.

    Args:
        aws_profile: str
            The AWS profile to get access to the EC2 instance
        ec2_tags: DictConfig
            The EC2 tags to find the EC2 instance
        mlflow_port: int
            The port on which MLflow is running
        mlflow_experiment: str
            The name of the MLflow experiment
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

    # Get the public IP address of the current EC2 instance
    public_ip = response["Reservations"][0]["Instances"][0]["PublicIpAddress"]

    remote_tracking_uri = f"http://{public_ip}:{mlflow_port}/"

    mlflow.set_tracking_uri(remote_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)


@task(name="Load Data", retries=3, retry_delay_seconds=2)
def load_data(path: str) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    """Function to load the data.

    Args:
        path: str
            Path to the data
    Returns:
        X_train: pd.DataFrame
            Training data
        y_train: pd.DataFrame
            Training target
        X_test: pd.DataFrame
            Testing data
        y_test: pd.DataFrame
            Testing target
    """
    # open X_y_data.pickle
    with open(path, "rb") as f:
        X_train, y_train, X_test, y_test = pickle.load(f)

    return X_train, y_train, X_test, y_test


@task(name="Train Model", retries=3, retry_delay_seconds=2)
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> any:
    """Function to train the model.

    Args:
        X_train: pd.DataFrame
            Training data
        y_train: pd.DataFrame
            Training target
    Returns:
        ols: sm.OLS
            Trained model
    """
    ols = sm.OLS(y_train, X_train).fit()

    return ols


@task(name="Evaluate Model", retries=3, retry_delay_seconds=2)
def evaluate_model(y_test: pd.Series, predictions: pd.Series) -> (float, float, float):
    """Function to evaluate the model.

    Args:
        y_test: pd.Series
            Real values of the test set
        predictions: pd.Series
            Predictions for the test set
    Returns:
        mae: float
            Mean absolute error
        mse: float
            Mean squared error
        r2: float
            R-squared
    """

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return mae, mse, r2


def make_prefect_report(
    experiment_name: str,
    model_name: str,
    data_path: str,
    mae: float,
    mse: float,
    r2: float,
) -> None:
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
    main_flow()
