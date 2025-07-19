#!/usr/bin/env python
# coding: utf-8
import os
import sys
import pickle
import pandas as pd


def read_data(path: str, categorical: list[str]) -> pd.DataFrame:
    endpoint_url = os.getenv("S3_ENDPOINT_URL")          
    storage_options = (
        {"client_kwargs": {"endpoint_url": endpoint_url}}
        if endpoint_url
        else None                                        
    )

    df = pd.read_parquet(path, storage_options=storage_options)
    return df

def prepare_data(df: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    df["duration"] = (
        df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    ).dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df = df[categorical + ["duration"]]

    df[categorical] = (
        df[categorical]
        .fillna(-1)
        .astype("int")
        .astype("str")
    )
    return df

def get_input_path(year: int, month: int) -> str:
    default = (
        "https://d37ci6vzurychx.cloudfront.net/trip-data/"
        "yellow_tripdata_{year:04d}-{month:02d}.parquet"
    )
    pattern = os.getenv("INPUT_FILE_PATTERN", default)
    return pattern.format(year=year, month=month)


def get_output_path(year: int, month: int) -> str:
    default = (
        "s3://nyc-duration-prediction-alexey/"
        "taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet"
    )
    pattern = os.getenv("OUTPUT_FILE_PATTERN", default)
    return pattern.format(year=year, month=month)

def main(year: int, month: int, categorical: list[str]) -> pd.DataFrame:
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    df = read_data(input_file, categorical)
    df = prepare_data(df, categorical)
    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    with open("model.bin", "rb") as f_in:
        dv, lr = pickle.load(f_in)

    X_val = dv.transform(df[categorical].to_dict(orient="records"))
    y_pred = lr.predict(X_val)

    print("predicted mean duration:", y_pred.mean())

    df_result = pd.DataFrame(
        {"ride_id": df["ride_id"], "predicted_duration": y_pred}
    )

    # same S3-vs-LocalStack trick for writing
    endpoint_url = os.getenv("S3_ENDPOINT_URL")
    storage_options = (
        {"client_kwargs": {"endpoint_url": endpoint_url}}
        if endpoint_url
        else None
    )
    df_result.to_parquet(output_file, engine="pyarrow",
                         index=False, storage_options=storage_options)

    print(f"results are saved to {output_file}")
    return df_result

    

if __name__ == "__main__":
    yr = int(sys.argv[1])
    mo = int(sys.argv[2])
    df_res = main(yr, mo, categorical=["PULocationID", "DOLocationID"])
    print(df_res.head())

