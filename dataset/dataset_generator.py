from datetime import date
from dateutil.relativedelta import relativedelta
from urllib.error import HTTPError

import pandas as pd
import os
import urllib.request
import zipfile

from dataset.data_utils import initialiaze_dataset


"""
Generate dataset for the provided symbols & the timeframe.

make:
    Args:
       symbols: List of symbols (for which dataset would be created).
       timeframe: data's timeframe (e.g., '1d', '1h').

    Returns:
       List of DataFrames for created datasets.
"""


def make(symbols, timeframe):
    print("Creating dataset", flush=True)
    datasets = []  # List to store the created DataFrames
    start_year = 2020  # Start year for data retrieval
    period_end = date.today() - relativedelta(months=1)  # End period for data retrieval
    end_year = period_end.year  # End year
    end_month = period_end.month  # End month

    # Loop through each symbol to create datasets
    for symbol in symbols:
        df = pd.DataFrame()
        dataset_destination = (
            f"dataset/{symbol}-{timeframe}-{start_year}_{end_year}.csv"
        )

        # Check if the dataset already exists
        if os.path.isfile(dataset_destination):
            print(f"{dataset_destination} already exists \n")
            continue
        print(f"Creating {symbol} {timeframe} dataset")

        # Create a range of periods for data retrieval
        pr = pd.period_range(
            start=f"{start_year}-01",
            end=f"{end_year}-{end_month}",
            freq="M",
        )

        # Convert the period range into tuples of (month, year)
        prTuples = tuple([(period.month, period.year) for period in pr])

        # Loop through each month and year tuple to download and process data
        for month, year in prTuples:
            monthly_dataset_destination_zip = (
                f"dataset/{symbol}-{timeframe}-{year}-{month:02d}.zip"
            )
            monthly_dataset_destination_csv = (
                f"dataset/{symbol}-{timeframe}-{year}-{month:02d}.csv"
            )
            try:
                # Check if the monthly CSV file exists
                if not os.path.isfile(monthly_dataset_destination_csv):
                    # Check if the monthly ZIP file exists
                    if not os.path.isfile(monthly_dataset_destination_zip):
                        print(
                            f"Downloading {monthly_dataset_destination_zip}", flush=True
                        )
                        # Download the ZIP file from Binance's data source
                        url = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/{timeframe}/{symbol}-{timeframe}-{year}-{month:02d}.zip"
                        urllib.request.urlretrieve(url, monthly_dataset_destination_zip)

                    # Extract the downloaded ZIP file
                    with zipfile.ZipFile(
                        monthly_dataset_destination_zip, "r"
                    ) as zip_ref:
                        zip_ref.extractall("dataset")

                    # Remove the ZIP file after extraction
                    if os.path.isfile(monthly_dataset_destination_zip):
                        os.remove(monthly_dataset_destination_zip)

                # Read the extracted CSV file into a DataFrame
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

                # Concatenate the monthly dataset to the main DataFrame
                df = pd.concat([df, monthly_dataset], ignore_index=True)

                # Remove the monthly CSV file after processing
                if os.path.isfile(monthly_dataset_destination_csv):
                    os.remove(monthly_dataset_destination_csv)

            # Handle the case where the data does not exist on Binance
            except HTTPError as e:
                print(
                    f"Error {str(e)}, {year}-{month} does not exist on Binance, continuing"
                )

        # Remove any duplicate header rows in the DataFrame
        df = df.drop(df[df["open_time"] == "open_time"].index).reset_index(drop=True)
        print("Processing dataset... \n", flush=True)

        # Initialize the dataset
        initialiaze_dataset(df)

        # Check if the full dataset already exists
        if os.path.isfile(dataset_destination):
            print(f"{dataset_destination} already exists, aborting saving in csv")
        else:
            # Save the processed DataFrame to a CSV file
            df.to_csv(dataset_destination)

        # Append the DataFrame to the datasets list
        datasets.append(df)
