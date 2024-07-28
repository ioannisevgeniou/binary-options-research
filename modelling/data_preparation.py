from sklearn.model_selection import train_test_split
from numpy import loadtxt, savetxt

"""
    Conducting data preprocessing and splitting using the inputted dataframe.

    Args:
        df (DataFrame): Dataframe that is being inputted for processing.

    Returns:
        tuple: Six elements - X_train, y_train, X_valid, y_valid, X_test, y_test.

"""


def split_data(df):
    df = df.dropna().reset_index(drop=True)

    for col in df.select_dtypes(include="bool").columns:
        df[col] = df[col].astype(int)

    X = df.drop(
        columns=[
            "result",
            "pattern",
            "open_time_start",
            "open_time_reversal1",
            "open_time_msb",
            "open_time_reversal2",
            "open_time_end",
        ]
    )
    y = df["result"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=0.8, random_state=42
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, train_size=0.5, random_state=42
    )

    return X_train, y_train, X_valid, y_valid, X_test, y_test


"""
    The datasets viz; the training, the validation, and the test are being saved into the CSV files.

    Args:
        X_train : The DataFrame exhibiting the training features.
        y_train : The series exhibiting the training targets.
        X_valid : The DataFrame exhibiting the validation features.
        y_valid : The series exhibiting the validation targets.
        X_test  : The dataframe exhibiting the test features.
        y_test  : The series exhibiting the test targets.

"""


def save_data(X_train, y_train, X_valid, y_valid, X_test, y_test):
    savetxt("modelling/X_train.csv", X_train, delimiter=",", fmt="%.3f")
    savetxt("modelling/y_train.csv", y_train, delimiter=",", fmt="%i")
    savetxt("modelling/X_valid.csv", X_valid, delimiter=",", fmt="%.3f")
    savetxt("modelling/y_valid.csv", y_valid, delimiter=",", fmt="%i")
    savetxt("modelling/X_test.csv", X_test, delimiter=",", fmt="%.3f")
    savetxt("modelling/y_test.csv", y_test, delimiter=",", fmt="%i")


"""
    The datasets viz; the training, the validation and the test are being loaded from the CSV.

    Returns:
        tuple: Six elements - X_train, y_train, X_valid, y_valid, X_test, y_test.
"""


def load_data():
    X_train = loadtxt("modelling/X_train.csv", delimiter=",")
    y_train = loadtxt("modelling/y_train.csv", delimiter=",")
    X_valid = loadtxt("modelling/X_valid.csv", delimiter=",")
    y_valid = loadtxt("modelling/y_valid.csv", delimiter=",")
    X_test = loadtxt("modelling/X_test.csv", delimiter=",")
    y_test = loadtxt("modelling/y_test.csv", delimiter=",")

    return X_train, y_train, X_valid, y_valid, X_test, y_test
