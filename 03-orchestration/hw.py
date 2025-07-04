from prefect import flow, task, get_run_logger
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import pandas as pd
import mlflow
import mlflow.sklearn

@task
def load_data(url: str) -> pd.DataFrame:
    df = pd.read_parquet(url)
    logger = get_run_logger()
    logger.info (f"Number of records loaded: {len(df)}")
    return df

def read_dataframe(url: str) -> pd.DataFrame:
    df = pd.read_parquet(url)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    logger = get_run_logger()
    logger.info (f"Number of records loaded: {len(df)}")

    return df

def train_model (df):
    logger = get_run_logger()
    
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    target = 'duration'
    dv = DictVectorizer()
    train_dicts = df[categorical].to_dict(orient='records') 
    
    X_train = dv.fit_transform(train_dicts)
    y_train = df[target].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    logger.info(f"Model intercept: {model.intercept_}")

    mlflow.set_experiment("prefect-taxi-experiment")
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, artifact_path="model")
    return dv, model

@flow
def taxi_data_pipeline():
    url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
    df = read_dataframe(url)
    dv, model = train_model(df)

if __name__ == "__main__":
    taxi_data_pipeline()