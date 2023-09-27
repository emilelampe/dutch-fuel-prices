"""This script processes the data so that it can be used by the ML model.
It downloads data from CBS and merges it with the oil prices and the EUR/USD exchange rate.
The data is then transformed and additional features are created.
A pickle file with the processed data is saved.
The file contains X_train, y_train, X_test, and y_test.

Author: Emile Lampe
"""

import datetime
import os
import pickle

import cbsodata
import hydra
import pandas as pd
from omegaconf import DictConfig

DAYS_OF_WEEK = {
    "zondag": "Sunday",
    "maandag": "Monday",
    "dinsdag": "Tuesday",
    "woensdag": "Wednesday",
    "donderdag": "Thursday",
    "vrijdag": "Friday",
    "zaterdag": "Saturday",
}

MONTHS = {
    "januari": "January",
    "februari": "February",
    "maart": "March",
    "april": "April",
    "mei": "May",
    "juni": "June",
    "juli": "July",
    "augustus": "August",
    "september": "September",
    "oktober": "October",
    "november": "November",
    "december": "December",
}


@hydra.main(config_path="../config", config_name="main", version_base=None)
def process_data(config: DictConfig):
    """Function to process the data.
    After the data is processed, it is saved as a pickle file.

    Args:
        None

    Returns:
        None
    """

    conversions_path = config.data.raw.eur_usd
    oil_path = config.data.raw.oil_prices

    conversions = pd.read_csv(conversions_path)
    oil = pd.read_csv(oil_path)

    target = config.process.target
    features = config.process.features

    names_dict = config.process.names
    cbs_dict = config.process.cbs
    oil_conversions_dict = config.process.oil_conv
    days_dict = config.process.days
    suffix_dict = config.process.suffixes

    date_name = names_dict.date
    first_training_date = config.process.first_training_date
    first_training_date = datetime.datetime.strptime(
        first_training_date, "%Y-%m-%d"
    ).date()

    save_data_location = config.data.processed

    print(f"Process data using {conversions_path} and {oil_path}")
    print(f"Target: {target}")
    print(f"Features: {features}")

    oil_conv_df = merge_oil_conversions(
        oil=oil,
        conversions=conversions,
        names_dict=names_dict,
        oil_conv=oil_conversions_dict,
    )

    fuel, _ = get_cbs_data(cbs_dict=cbs_dict)
    fuel = process_cbs_data(
        df=fuel, cbs_dict=cbs_dict, names_dict=names_dict, new_target_name=target
    )

    df = pd.merge(oil_conv_df, fuel, on=date_name, how="outer").sort_index(
        ascending=True
    )

    begin_date, end_date = get_test_dates(days_dict=days_dict)

    X, y = transform(
        df=df,
        target=target,
        features=features,
        names=names_dict,
        days_dict=days_dict,
        suffix_dict=suffix_dict,
    )

    X_train, y_train, X_test, y_test = split_train_test(
        X=X,
        y=y,
        begin=begin_date,
        end=end_date,
        first_training_date=first_training_date,
        days_dict=days_dict,
    )

    with open(save_data_location, "wb") as f:
        pickle.dump((X_train, y_train, X_test, y_test), f)


def merge_oil_conversions(
    oil: pd.DataFrame, conversions: pd.DataFrame, names_dict, oil_conv
) -> pd.DataFrame:
    """Function to merge the oil prices with the EUR/USD exchange rate.

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
    """

    date_name = names_dict.date
    new_oil_eur = names_dict.oil_eur
    new_oil_usd = names_dict.oil_usd
    new_eur_usd = names_dict.eur_usd

    old_eur_usd = oil_conv.names.eur_usd
    old_oil_usd = oil_conv.names.oil_usd

    columns_to_keep = oil_conv.columns

    df = pd.merge(
        oil,
        conversions.add_suffix("_USD"),
        left_on=date_name,
        right_on=date_name + "_USD",
    )
    df.Date = pd.to_datetime(df.Date)
    df = df.rename(columns={old_eur_usd: new_eur_usd, old_oil_usd: new_oil_usd})
    df[new_oil_eur] = df[new_oil_usd] / df[new_eur_usd]
    df = df[columns_to_keep]
    df[new_oil_eur] = df[new_oil_eur].apply(lambda x: round(x, 2))
    df.set_index(date_name, inplace=True)
    df.sort_index(ascending=True, inplace=True)

    return df


def get_cbs_data(cbs_dict):
    """Function to get the data from CBS.
    Uses cache if available.

    Args:
        cbs_dict: DictConfig
            The configuration of the CBS data

    Returns:
        ds: pd.DataFrame
            The dataframe with the CBS data
        columninfo: pd.DataFrame
            The dataframe with the column information of the CBS data
    """

    cbs_code = cbs_dict.code
    cache = cbs_dict.cache
    cache_dir = cbs_dict.cache_dir
    data_properties_name = cbs_dict.names.data_properties

    cache_file = os.path.join(cache_dir, cbs_code + ".pickle")
    if cache and os.path.exists(cache_file):
        ds = pd.read_pickle(cache_file)
    else:
        ds = pd.DataFrame(cbsodata.get_data(cbs_code))
        for c in ds.columns:
            if ds[c].dtype.str == "|O":
                ds[c] = ds[c].str.strip()
        ds.to_pickle(cache_file)

    cache_file2 = os.path.join(cache_dir, cbs_code + "_info.pickle")
    if cache and os.path.exists(cache_file2):
        columninfo = pd.read_pickle(cache_file2)
    else:
        columninfo = pd.DataFrame(cbsodata.get_meta(cbs_code), data_properties_name)
        columninfo.to_pickle(cache_file2)
    return ds, columninfo


def process_cbs_data(df, cbs_dict, names_dict, new_target_name):
    """Function to process the CBS data.

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
    """

    drop_columns = cbs_dict.drop_columns
    cbs_date_name = cbs_dict.names.date
    cbs_target_name = cbs_dict.names.target

    new_date_name = names_dict.date

    # replace any value of day_of_week and months with the english version in df["Perioden"]
    df[cbs_date_name] = df[cbs_date_name].replace(DAYS_OF_WEEK, regex=True)
    df[cbs_date_name] = df[cbs_date_name].replace(MONTHS, regex=True)

    df[cbs_date_name] = pd.to_datetime(df[cbs_date_name])

    df.drop(drop_columns, axis=1, inplace=True)

    df.rename(
        columns={cbs_target_name: new_target_name, cbs_date_name: new_date_name},
        inplace=True,
    )
    df.set_index(new_date_name, inplace=True)

    return df


def get_test_dates(days_dict):
    """Function to get the dates of the test set.

    Args:
        days_dict: DictConfig
            The configuration of the days

    Returns:
        begin: datetime.date
            The first date of the test set
        end: datetime.date
            The last date of the test set
    """

    days_to_predict = days_dict.predict
    long_lag = days_dict.long_lag

    today = datetime.date.today()
    idx = (today.weekday() + 1) % 7
    # to make sure you get enough time to predict the next 14 days and skip the furthest lag of smoothing
    # 7 is added because it takes a week before the data is uploaded to CBS
    begin = today - datetime.timedelta(idx + days_to_predict + long_lag - 1)
    # to predict until the Sunday available on the website
    end = today - datetime.timedelta(idx + 7)

    return begin, end


def transform(df, target, features, names, days_dict, suffix_dict):
    """Function to transform the data.
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
    """

    days_to_predict = days_dict.predict
    rolling_days = days_dict.rolling
    short_lag = days_dict.short_lag

    smoothed_name = suffix_dict.smoothed
    short_lag_name = suffix_dict.short_lag
    long_lag_name = suffix_dict.long_lag
    delta_name = suffix_dict.delta

    oil_name = names.oil_eur
    eu95_name = names.eu95
    eur_usd_name = names.eur_usd

    df = df.resample("1d").max().interpolate()
    df.dropna(inplace=True)
    df[oil_name + smoothed_name] = df[oil_name].rolling(rolling_days).mean()
    df[oil_name + smoothed_name].apply(lambda x: round(x, 3))
    df[eu95_name + smoothed_name] = df[eu95_name].rolling(rolling_days).mean()
    df[eu95_name + smoothed_name].apply(lambda x: round(x, 3))
    df[eur_usd_name].apply(lambda x: round(x, 3))

    df[oil_name + short_lag_name] = df[oil_name + smoothed_name].shift(days_to_predict)
    df[oil_name + long_lag_name] = df[oil_name + smoothed_name].shift(
        short_lag + days_to_predict
    )
    df[oil_name + delta_name] = (
        df[oil_name + long_lag_name] - df[oil_name + short_lag_name]
    )

    df[eu95_name + short_lag_name] = df[eu95_name + smoothed_name].shift(
        days_to_predict
    )
    df[eu95_name + long_lag_name] = df[eu95_name + smoothed_name].shift(
        short_lag + days_to_predict
    )
    df[eu95_name + delta_name] = (
        df[eu95_name + long_lag_name] - df[eu95_name + short_lag_name]
    )

    features = features
    target = target

    X = df[features]
    y = df[target]

    return X, y


def split_train_test(X, y, begin, end, first_training_date, days_dict):
    """Function to split the data into a train and test set.

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
    """

    rolling_days = days_dict.rolling

    X_train = X[: begin - datetime.timedelta(days=1)]
    y_train = y[: begin - datetime.timedelta(days=1)]
    X_train = X_train.loc[first_training_date:]
    y_train = y_train.loc[first_training_date:]

    X_test = X[begin + datetime.timedelta(days=rolling_days) : end]
    y_test = y[begin + datetime.timedelta(days=rolling_days) : end]

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    process_data()
