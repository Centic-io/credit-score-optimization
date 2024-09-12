from datetime import datetime
import pandas as pd
import numpy as np


def custom_normalization(
    df, column, min_threshold=None, max_threshold=None, zero=True, reverse=False
):
    def transform_score(value, min_val, max_val, reverse):
        if value <= min_val:
            return 850 if reverse else 300
        elif value >= max_val:
            return 300 if reverse else 850
        else:
            scale = (max_val - value) if reverse else (value - min_val)
            return (scale / (max_val - min_val)) * 550 + 300

    df_filtered = df[df[column] != 0] if zero else df
    min_val = (
        df_filtered[column].quantile(min_threshold) if min_threshold is not None else 0
    )
    max_val = df_filtered[column].quantile(max_threshold)

    df[column] = df[column].apply(transform_score, args=(min_val, max_val, reverse))
    return df


def read_data():
    df = pd.read_csv("./data/Lending-Data-Ethereum-Labeled.csv")
    df = df.dropna()
    df = df[df["depositInUSD"] > df["borrowInUSD"]]
    df = df[df["totalAsset"] >= 10]
    df.loc[df["borrowInUSD"] < 0.005, "borrowInUSD"] = 0
    df.loc[df["totalAsset"] < 0.005, "totalAsset"] = 0
    df.loc[df["depositInUSD"] < 0.005, "depositInUSD"] = 0
    df.loc[df["balanceInUSD"] < 0.005, "balanceInUSD"] = 0.0
    df["borrow_per_balance"] = np.where(
        df["balanceInUSD"] == 0, 0, df["borrowInUSD"] / df["balanceInUSD"]
    )
    df["borrow_per_deposit"] = np.where(
        df["depositInUSD"] == 0, 0, df["borrowInUSD"] / df["depositInUSD"]
    )
    df["averageTotalAsset"] = (
        df["averageBalance"] + df["depositInUSD"] - df["borrowInUSD"]
    )
    df["deposit_per_asset"] = np.where(
        df["totalAsset"] == 0, 0, df["depositInUSD"] / df["totalAsset"]
    )
    current_timestamp = int(datetime.now().timestamp())
    df["age"] = (current_timestamp - df["createdAt"]).astype(int)

    # Drop column
    df_normalized = (
        df.drop("depositInUSD", axis=1)
        .drop("borrowInUSD", axis=1)
        .drop("balanceInUSD", axis=1)
        .drop("createdAt", axis=1)
        .drop("averageBalance", axis=1)
        .drop("address", axis=1)
        # .drop("numberOfReputableDapps", axis=1)
    )

    df_normalized = df_normalized[
        ~(
            (df_normalized["frequencyOfTransaction"] == 0)
            & (df_normalized["frequencyMountOfTransaction"] > 0)
        )
    ]
    df_normalized = df_normalized[df_normalized >= 0]
    main_label_column = df["1st_label"]
    sub_label_column = df["2nd_label"]

    # totalAsset
    df_normalized = custom_normalization(
        df=df_normalized,
        column="totalAsset",
        zero=False,
        # min_threshold=0.25,
        max_threshold=0.95,
        reverse=False,
    )
    # averageTotalAsset
    df_normalized = custom_normalization(
        df=df_normalized,
        column="averageTotalAsset",
        zero=False,
        # min_threshold=0.25,
        max_threshold=0.95,
        reverse=False,
    )

    # frequencyOfDappTransactions
    df_normalized = custom_normalization(
        df=df_normalized,
        column="frequencyOfDappTransactions",
        zero=True,
        max_threshold=0.95,
        reverse=False,
    )
    # numberOfInteractedDapps
    df_normalized = custom_normalization(
        df=df_normalized,
        column="numberOfInteractedDapps",
        zero=True,
        max_threshold=0.95,
        reverse=False,
    )
    # typesOfInteractedDapps
    df_normalized = custom_normalization(
        df=df_normalized,
        column="typesOfInteractedDapps",
        zero=True,
        max_threshold=0.95,
        reverse=False,
    )
    # numberOfReputableDapps
    df_normalized = custom_normalization(
        df=df_normalized,
        column="numberOfReputableDapps",
        zero=True,
        max_threshold=0.99,
        reverse=False,
    )

    # frequencyMountOfTransaction
    df_normalized = custom_normalization(
        df=df_normalized,
        column="frequencyMountOfTransaction",
        zero=True,
        # min_threshold=0.24,
        max_threshold=0.99,
        reverse=False,
    )
    # frequencyOfTransaction
    df_normalized = custom_normalization(
        df=df_normalized,
        column="frequencyOfTransaction",
        zero=True,
        max_threshold=0.98,
        reverse=False,
    )
    # age
    df_normalized = custom_normalization(
        df=df_normalized,
        column="age",
        zero=False,
        min_threshold=0.05,
        max_threshold=0.95,
        reverse=False,
    )
    # numberOfLiquidation
    df_normalized = custom_normalization(
        df=df_normalized,
        column="numberOfLiquidation",
        zero=True,
        max_threshold=0.98,
        reverse=True,
    )
    # totalValueOfLiquidation
    df_normalized = custom_normalization(
        df=df_normalized,
        column="totalValueOfLiquidation",
        zero=True,
        max_threshold=0.77,
        reverse=True,
    )
    # borrow_per_balance
    df_normalized = custom_normalization(
        df=df_normalized,
        column="borrow_per_balance",
        zero=True,
        # min_threshold=0.36,
        max_threshold=0.95,
        reverse=True,
    )
    # borrow_per_deposit
    df_normalized = custom_normalization(
        df=df_normalized,
        column="borrow_per_deposit",
        zero=True,
        min_threshold=0.05,
        max_threshold=0.95,
        reverse=True,
    )
    # deposit_per_asset
    df_normalized = custom_normalization(
        df=df_normalized,
        column="deposit_per_asset",
        zero=False,
        # min_threshold=0.3,
        max_threshold=0.95,
        reverse=False,
    )

    df_normalized["1st_label"] = main_label_column
    df_normalized["2nd_label"] = sub_label_column
    return df_normalized


def read_data_without_nomalize():
    df = pd.read_csv("./data/Lending-Data-Ethereum-Labeled.csv")
    df = df.dropna()
    df = df[df["depositInUSD"] > df["borrowInUSD"]]
    df = df[df["totalAsset"] >= 10]
    df.loc[df["borrowInUSD"] < 0.005, "borrowInUSD"] = 0.0
    df.loc[df["totalAsset"] < 0.005, "totalAsset"] = 0.0
    df.loc[df["depositInUSD"] < 0.005, "depositInUSD"] = 0.0
    df.loc[df["balanceInUSD"] < 0.005, "balanceInUSD"] = 0.0
    df["borrow_per_balance"] = np.where(
        df["balanceInUSD"] == 0, 0, df["borrowInUSD"] / df["balanceInUSD"]
    )
    df["borrow_per_deposit"] = np.where(
        df["depositInUSD"] == 0, 0, df["borrowInUSD"] / df["depositInUSD"]
    )
    df["averageTotalAsset"] = (
        df["averageBalance"] + df["depositInUSD"] - df["borrowInUSD"]
    )
    df["deposit_per_asset"] = np.where(
        df["totalAsset"] == 0, 0, df["depositInUSD"] / df["totalAsset"]
    )
    current_timestamp = int(datetime.now().timestamp())
    df["age"] = (current_timestamp - df["createdAt"]).astype(int)

    # Drop column
    df_normalized = (
        df.drop("address", axis=1)
        .drop("balanceInUSD", axis=1)
        .drop("depositInUSD", axis=1)
        .drop("borrowInUSD", axis=1)
        .drop("createdAt", axis=1)
        .drop("averageBalance", axis=1)
        # .drop("numberOfReputableDapps", axis=1)
    )

    df_normalized = df_normalized[
        ~(
            (df_normalized["frequencyOfTransaction"] == 0)
            & (df_normalized["frequencyMountOfTransaction"] > 0)
        )
    ]
    df_normalized = df_normalized[df_normalized >= 0]
    return df_normalized
