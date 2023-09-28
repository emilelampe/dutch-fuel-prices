Module src.process
==================
This script processes the data so that it can be used by the ML model.
It downloads data from CBS and merges it with the oil prices and the EUR/USD exchange rate.
The data is then transformed and additional features are created.
A pickle file with the processed data is saved.
The file contains X_train, y_train, X_test, and y_test.

Author: Emile Lampe

Functions
---------

    
`get_cbs_data(cbs_dict)`
:   Function to get the data from CBS.
    Uses cache if available.
    
    Args:
        cbs_dict: DictConfig
            The configuration of the CBS data
    
    Returns:
        ds: pd.DataFrame
            The dataframe with the CBS data
        columninfo: pd.DataFrame
            The dataframe with the column information of the CBS data

    
`get_test_dates(days_dict)`
:   Function to get the dates of the test set.
    
    Args:
        days_dict: DictConfig
            The configuration of the days
    
    Returns:
        begin: datetime.date
            The first date of the test set
        end: datetime.date
            The last date of the test set

    
`merge_oil_conversions(oil: pandas.core.frame.DataFrame, conversions: pandas.core.frame.DataFrame, names_dict, oil_conv) ‑> pandas.core.frame.DataFrame`
:   Function to merge the oil prices with the EUR/USD exchange rate.
    
    Args:
        oil: pd.DataFrame
            The dataframe with the oil prices
        conversions: pd.DataFrame
            The dataframe with the EUR/USD exchange rate
        names_dict: DictConfig
            The names of the columns in the dataframes
        oil_conv: DictConfig
            The names of the columns in the dataframes
    
    Returns:
        df: pd.DataFrame
            The dataframe with the oil prices and the EUR/USD exchange rate

    
`process_cbs_data(df, cbs_dict, names_dict, new_target_name)`
:   Function to process the CBS data.
    
    Args:
        df: pd.DataFrame
            The dataframe with the CBS data
        cbs_dict: DictConfig
            The configuration of the CBS data
        names_dict: DictConfig
            The names of the columns in the dataframes
    
    Returns:
        df: pd.DataFrame
            The dataframe with the processed CBS data

    
`process_data(config: omegaconf.dictconfig.DictConfig)`
:   Function to process the data.
    After the data is processed, it is saved as a pickle file.
    
    Args:
        None
    
    Returns:
        None

    
`split_train_test(X, y, begin, end, first_training_date, days_dict)`
:   Function to split the data into a train and test set.
    
    Args:
        X: pd.DataFrame
            The data of the features
        y: pd.DataFrame
            The data of the target
        begin: datetime.date
            The first date of the test set
        end: datetime.date
            The last date of the test set
        first_training_date: datetime.date
            The first date of the training set
        days_dict: DictConfig
            The configuration of the days
    
    Returns:
        X_train: pd.DataFrame
            The data of the features of the training set
        y_train: pd.DataFrame
            The data of the target of the training set
        X_test: pd.DataFrame
            The data of the features of the test set
        y_test: pd.DataFrame
            The data of the target of the test set

    
`transform(df, target, features, names, days_dict, suffix_dict)`
:   Function to transform the data.
    Includes the smoothing of the oil prices and the EUR/USD exchange rate.
    Includes the creation of the short lag and long lag.
    
    Args:
        df: pd.DataFrame
            The dataframe with the data
        target: str
            The name of the target
        features: list
            The names of the features
        names: DictConfig
            The names of the columns in the dataframes
        days_dict: DictConfig
            The configuration of the days
        suffix_dict: DictConfig
            The names of the suffixes
    
    Returns:
        X: pd.DataFrame
            The data of the features
        y: pd.DataFrame
            The data of the target