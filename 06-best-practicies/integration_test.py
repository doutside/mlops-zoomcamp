#!/usr/bin/env python
# coding: utf-8

import os
import subprocess
from datetime import datetime

import pandas as pd
from batch import get_input_path, get_output_path, prepare_data  # reuse helpers

import os, boto3, botocore

endpoint   = os.getenv("S3_ENDPOINT_URL")          
bucket_out = "nyc-duration-prediction-gizmo"

s3 = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
)

def ensure_bucket(name: str) -> None:
    try:
        s3.head_bucket(Bucket=name)
        print(f"✓ Bucket {name} already exists")
    except botocore.exceptions.ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("404", "NoSuchBucket", "NotFound"):
            print(f"⎔ Creating bucket {name}")
            s3.create_bucket(Bucket=name)
        else:
            raise                      

ensure_bucket(bucket_out)

# ------------------------------------------------------------------------------
# Local helpers
# ------------------------------------------------------------------------------
def dt(h: int, m: int, s: int = 0) -> datetime:
    return datetime(2023, 1, 1, h, m, s)


def build_raw_df() -> pd.DataFrame:
    data = [
        (None, None, dt(1, 1),  dt(1, 10)),
        (1,    1,    dt(1, 2),  dt(1, 10)),
        (1,    None, dt(1, 2),  dt(1, 2, 59)),
        (3,    4,    dt(1, 2),  dt(2, 2, 1)),
    ]
    cols = [
        "PULocationID",
        "DOLocationID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
    ]
    return pd.DataFrame(data, columns=cols)


def save_data(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to S3, honouring LocalStack endpoint if set."""
    endpoint = os.getenv("S3_ENDPOINT_URL")
    opts = {"client_kwargs": {"endpoint_url": endpoint}} if endpoint else None

    df.to_parquet(
        path,
        engine="pyarrow",
        compression=None,
        index=False,
        storage_options=opts,
    )


def load_parquet(path: str) -> pd.DataFrame:
    """Read a parquet file from S3 with the same LocalStack switch."""
    endpoint = os.getenv("S3_ENDPOINT_URL")
    opts = {"client_kwargs": {"endpoint_url": endpoint}} if endpoint else None
    return pd.read_parquet(path, storage_options=opts)


# ------------------------------------------------------------------------------
# 1 ▸ Seed S3 with the input parquet
# ------------------------------------------------------------------------------
year, month = 2023, 1
input_key = get_input_path(year, month)    # e.g. s3://nyc-duration/in/2023-01.parquet

raw_df = build_raw_df()
save_data(raw_df, input_key)
print(f"✓ Uploaded input data to {input_key}")

# ------------------------------------------------------------------------------
# 2 ▸ Run batch.py to generate predictions
# ------------------------------------------------------------------------------
print("▶ Running batch.py ...")
subprocess.check_call(["python", "batch.py", str(year), f"{month:02d}"])

# ------------------------------------------------------------------------------
# 3 ▸ Load the output and report the metric
# ------------------------------------------------------------------------------
output_key = get_output_path(year, month)  # same helper as inside batch.py
pred_df = load_parquet(output_key)

total_duration = pred_df["predicted_duration"].sum()
print(f"\nΣ predicted_duration for test DataFrame = {total_duration:.4f} minutes")
