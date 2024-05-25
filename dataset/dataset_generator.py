from datetime import date
from dateutil.relativedelta import relativedelta
from urllib.error import HTTPError

import pandas as pd
import os
import urllib.request
import zipfile

from dataset.data_utils import initialiaze_dataset


def make(symbols, timeframe):
    """
    Parameters
    ----------

    `combine` : Whether to create a single dataset of all `symbols` together
    `individuals` : Whether to create separate a dataset for each `symbol`

    `combine` and `individuals` can be used in any combination
    """

    print("Creating dataset")
    datasets = []
    start_year = 2020
    period_end = date.today() - relativedelta(months=1)
    end_year = period_end.year
    end_month = period_end.month
    for symbol in symbols:
        df = pd.DataFrame()
        dataset_destination = (
            f"dataset/{symbol}-{timeframe}-{start_year}_{end_year}.csv"
        )
        if os.path.isfile(dataset_destination):
            print(f"{dataset_destination} already exists")
            continue
        print(f"Creating {symbol} {timeframe} dataset")
        pr = pd.period_range(
            start=f"{start_year}-01",
            end=f"{end_year}-{end_month}",
            freq="M",
        )
        prTuples = tuple([(period.month, period.year) for period in pr])
        for month, year in prTuples:
            monthly_dataset_destination_zip = (
                f"dataset/{symbol}-{timeframe}-{year}-{month:02d}.zip"
            )
            monthly_dataset_destination_csv = (
                f"dataset/{symbol}-{timeframe}-{year}-{month:02d}.csv"
            )
            try:
                if not os.path.isfile(monthly_dataset_destination_csv):
                    if not os.path.isfile(monthly_dataset_destination_zip):
                        print(f"Downloading {monthly_dataset_destination_zip}")
                        url = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/{timeframe}/{symbol}-{timeframe}-{year}-{month:02d}.zip"
                        urllib.request.urlretrieve(url, monthly_dataset_destination_zip)

                    with zipfile.ZipFile(
                        monthly_dataset_destination_zip, "r"
                    ) as zip_ref:
                        zip_ref.extractall("dataset")

                    if os.path.isfile(monthly_dataset_destination_zip):
                        os.remove(monthly_dataset_destination_zip)

                monthly_dataset = pd.read_csv(
                    monthly_dataset_destination_csv,
                    names=[
                        "open_time",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "close_time",
                        "quote_asset_volume",
                        "number_of_trades",
                        "taker_buy_base_asset_volume",
                        "taker_buy_quote_asset_volume",
                        "ignore",
                    ],
                ).reset_index(drop=True)
                df = pd.concat([df, monthly_dataset], ignore_index=True)

                if os.path.isfile(monthly_dataset_destination_csv):
                    os.remove(monthly_dataset_destination_csv)
            except HTTPError as e:
                print(
                    f"Error {str(e)}, {year}-{month} does not exist on Binance, continuing"
                )

        # some candle had open_time as `open_time`, get rid of it
        df = df.drop(df[df["open_time"] == "open_time"].index).reset_index(drop=True)

        # Process dataset
        print("Processing dataset...")
        initialiaze_dataset(df)
        if os.path.isfile(dataset_destination):
            print(f"{dataset_destination} already exists, aborting saving in csv")
        else:
            df.to_csv(dataset_destination)
        datasets.append(df)
