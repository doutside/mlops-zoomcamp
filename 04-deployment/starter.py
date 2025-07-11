import argparse
import pickle
import pandas as pd
import os
import numpy as np
import sklearn
import sys

print(f"scikit-learn version: {sklearn.__version__}")
print(f"Python version: {sys.version.split()[0]}")

parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, default=2023, help="Year of the data")
parser.add_argument("--month", type=int, default=3, help="Month of the data")
args = parser.parse_args()

year = f"{args.year:04d}"
month = f"{args.month:02d}"
uri_name = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet'

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)
categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):

    df = pd.read_parquet(filename)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    print ('Data loaded successfully from', filename)
    return df

df = read_data(uri_name)

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

std_minutes = np.std(y_pred, ddof=1) 
print(f"Standard deviation of predicted duration: {std_minutes:.3f} minutes")

mean_minutes = np.mean(y_pred)
print(f"Mean predicted duration: {mean_minutes:.3f} minutes")

df['ride_id'] = f'{year}/{month}_' + df.index.astype('str')
df['preds'] = y_pred

df_result = pd.DataFrame({
         "ride_id": df["ride_id"],
         "predicted_duration": y_pred
     })

output_file = 'predictions.result.parquet'
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

file_mb = os.path.getsize(output_file) / 1024**2
print(f"local parquet file   : {file_mb:,.2f} MB")

