#!/usr/bin/env python
# coding: utf-8
# batch.py
import sys
import pickle
import pandas as pd
from datetime import datetime

def read_data(filename):
    df = pd.read_parquet(filename)
    return df

def prepare_data(df, categorical):    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def run(year, month):
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    categorical = ['PULocationID', 'DOLocationID']
    df = read_data(input_file)
    df = prepare_data(df, categorical)

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(output_file, engine='pyarrow', index=False)
    print(df_result)
    print(f'Results are saved to {output_file}')
    print('Done!')  

if __name__ == '__main__':
    year  = int(sys.argv[1])
    month = int(sys.argv[2])
    run(year, month)
