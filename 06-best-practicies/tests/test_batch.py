import pandas as pd
from batch import prepare_data
from datetime import datetime
import pandas.testing as pdt

def dt(h, m, s=0):
    return datetime(2023, 1, 1, h, m, s)

def test_prepare_data_filters_duration():
    data = [
        (None, None, dt(1, 1),       dt(1, 10)),
        (1,    1,    dt(1, 2),       dt(1, 10)),
        (1,    None, dt(1, 2, 0),    dt(1, 2, 59)),
        (3,    4,    dt(1, 2, 0),    dt(2, 2, 1)),
    ]
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    test_df = pd.DataFrame(data, columns=columns)

    actual = (
        prepare_data(test_df, ["PULocationID", "DOLocationID"])
        .reset_index(drop=True)           # make index comparable
        [["PULocationID", "DOLocationID", "duration"]]
    )

    print (actual)

    expected = pd.DataFrame(
        {
            "PULocationID": ["-1", "1"],  # NaNs → -1 → str
            "DOLocationID": ["-1", "1"],
            "duration": [9.0, 8.0],       # minutes
        }
    )

    pdt.assert_frame_equal(actual, expected, check_dtype=False)