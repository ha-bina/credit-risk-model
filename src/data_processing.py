# src/feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class TransactionTimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, time_col="TransactionStartTime"):
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col], errors='coerce')
        df["TransHour"] = df[self.time_col].dt.hour
        df["TransDay"] = df[self.time_col].dt.day
        df["TransWeekday"] = df[self.time_col].dt.weekday
        df["IsWeekend"] = df["TransWeekday"].isin([5, 6]).astype(int)
        df["TransMonth"] = df[self.time_col].dt.month
        return df.drop(columns=[self.time_col])


class TransactionAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, id_col="AccountId", amount_col="Amount"):
        self.id_col = id_col
        self.amount_col = amount_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["Credit"] = df[self.amount_col].apply(lambda x: abs(x) if x < 0 else 0)
        df["Debit"] = df[self.amount_col].apply(lambda x: x if x > 0 else 0)

        agg = df.groupby(self.id_col).agg({
            "Amount": ["sum", "mean", "count", "std"],
            "Credit": "sum",
            "Debit": "sum",
            "ProductId": pd.Series.nunique,
            "ChannelId": pd.Series.nunique,
            "ProviderId": pd.Series.nunique
        }).fillna(0)

        agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
        agg["CreditDebitRatio"] = agg["Credit_sum"] / (agg["Debit_sum"] + 1e-5)
        agg = agg.reset_index()

        return agg


def build_feature_pipeline(numeric_cols, categorical_cols):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])
    return ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ])
from sklearn.preprocessing import OneHotEncoder
import sklearn


def get_onehot_encoder():
    skl_version = tuple(map(int, sklearn.__version__.split('.')[:2]))
    if skl_version >= (1, 2):
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
def preprocess_transaction_data(df):
    # Use correct argument for OneHotEncoder depending on sklearn version
    get_onehot_encoder()
    skl_version = tuple(map(int, sklearn.__version__.split('.')[:2]))
    if skl_version >= (1, 2):
        encoder = OneHotEncoder(sparse_output=False)
    else:
        encoder = OneHotEncoder(sparse=False)
    # ... rest of your code ...
    # Step 1: Extract time features
    time_pipe = TransactionTimeFeatures()
    df = time_pipe.transform(df)

    # Step 2: Aggregate behavioral features
    agg_pipe = TransactionAggregator(id_col="AccountId", amount_col="Amount")
    agg_df = agg_pipe.transform(df)

    # Step 3: Merge static columns (categorical info per customer)
    static_cols = ["AccountId", "CurrencyCode", "CountryCode", "ProductCategory", "PricingStrategy", "ChannelId"]
    df_static = df[static_cols].drop_duplicates("AccountId")

    full = pd.merge(agg_df, df_static, on="AccountId", how="left")

    # Step 4: Define feature pipeline
    numeric_cols = [
        "Amount_sum", "Amount_mean", "Amount_count", "Amount_std",
        "Credit_sum", "Debit_sum", "ProductId_nunique", "ChannelId_nunique",
        "ProviderId_nunique", "CreditDebitRatio"
    ]
    categorical_cols = ["CurrencyCode", "CountryCode", "ProductCategory", "PricingStrategy", "ChannelId"]

    pipe = build_feature_pipeline(numeric_cols, categorical_cols)

    return pipe.fit_transform(full)
def save_processed_data(df, save_path):
   
    df.to_csv(save_path, index=False)
    print(f"Processed data saved to {save_path}")