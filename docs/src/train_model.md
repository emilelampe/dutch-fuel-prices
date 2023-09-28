Module src.train_model
======================
This is script trains the model using the data that is processed in process.py.

It has the following functions:
    * main_flow - function from which the flow starts
    * configure_mlflow - configures MLflow on the EC2 instance
    * load_data - loads the processed data
    * train_model - trains the model
    * evaluate_model - returns the scores of the model
    * make_prefect_report - makes a report of the results for Prefect

Author: Emile Lampe

Functions
---------

    
`configure_mlflow(aws_profile: str, ec2_tags: omegaconf.dictconfig.DictConfig, mlflow_port: int, mlflow_experiment: str) ‑> None`
:   Function to configure MLflow on the EC2 instance.
    
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

    
`evaluate_model(y_test: pandas.core.series.Series, predictions: pandas.core.series.Series) ‑> (<class 'float'>, <class 'float'>, <class 'float'>)`
:   Function to evaluate the model.
    
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

    
`load_data(path: str) ‑> (<class 'pandas.core.frame.DataFrame'>, <class 'pandas.core.series.Series'>, <class 'pandas.core.frame.DataFrame'>, <class 'pandas.core.series.Series'>)`
:   Function to load the data.
    
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

    
`main_flow(config: omegaconf.dictconfig.DictConfig)`
:   Function from which the flow starts.
    
    Args:
        None
    Returns:
        None

    
`make_prefect_report(experiment_name: str, model_name: str, data_path: str, mae: float, mse: float, r2: float) ‑> None`
:   

    
`train_model(X_train: pandas.core.frame.DataFrame, y_train: pandas.core.series.Series) ‑> <built-in function any>`
:   Function to train the model.
    
    Args:
        X_train: pd.DataFrame
            Training data
        y_train: pd.DataFrame
            Training target
    Returns:
        ols: sm.OLS
            Trained model