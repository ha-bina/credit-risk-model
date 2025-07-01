# src/proxy_target_engineering.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def calculate_rfm(df, customer_id_col="CustomerId", amount_col="Amount", date_col="TransactionStartTime", snapshot_date=None):
    """
    Calculates RFM (Recency, Frequency, Monetary) for each customer
    """
    df[date_col] = pd.to_datetime(df[date_col])

    # Define snapshot date (latest transaction + buffer if not given)
    if snapshot_date is None:
        snapshot_date = df[date_col].max() + pd.Timedelta(days=1)

    # Aggregate RFM metrics
    rfm = df.groupby(customer_id_col).agg({
        date_col: lambda x: (snapshot_date - x.max()).days,
        customer_id_col: "count",
        amount_col: lambda x: x[x > 0].sum()
    }).rename(columns={
        date_col: "Recency",
        customer_id_col: "Frequency",
        amount_col: "Monetary"
    }).reset_index()

    return rfm


def cluster_customers_rfm(rfm_df, n_clusters=3, random_state=42):
    """
    Cluster customers based on scaled RFM features using KMeans
    """
    rfm_values = rfm_df[["Recency", "Frequency", "Monetary"]]
    scaler = StandardScaler()
    scaled_rfm = scaler.fit_transform(rfm_values)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm_df["Cluster"] = kmeans.fit_predict(scaled_rfm)

    return rfm_df


def label_high_risk_customers(rfm_df):
    """
    Assign high risk to cluster with lowest engagement
    (High recency, low frequency, low monetary)
    """
    cluster_profiles = rfm_df.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    
    # High-risk = cluster with highest Recency and lowest Frequency + Monetary
    high_risk_cluster = cluster_profiles.sort_values(by=["Recency", "Frequency", "Monetary"], ascending=[False, True, True]).index[0]

    rfm_df["is_high_risk"] = (rfm_df["Cluster"] == high_risk_cluster).astype(int)
    return rfm_df.drop(columns=["Cluster"])


def create_proxy_target(df):
    """
    Main wrapper: input transaction dataframe, output high-risk labels per customer
    """
    rfm = calculate_rfm(df)
    clustered = cluster_customers_rfm(rfm)
    labeled = label_high_risk_customers(clustered)
    return labeled[["CustomerId", "is_high_risk"]]
