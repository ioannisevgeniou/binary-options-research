import numpy as np
import xgboost as xgb

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump

# Set random seed for reproducibility
np.random.seed(42)


"""
    Defines and trains an XGBoost classifier using hyper-parameters and 
    evaluates it on the validation set.

    Args:
        space (dict): hyperparameters and data containing dict for training and validation.

    Returns:
        dict: negative accuracy loss and status containing dict.
"""


def objective(space):
    # Initialize the XGBoost classifier with given hyperparameters
    model = xgb.XGBClassifier(
        n_jobs=-1,
        random_state=42,
        booster="gbtree",
        objective="binary:logistic",
        use_label_encoder=False,
        base_score=space["base_score"],
        scale_pos_weight=space["scale_pos_weight"],
        learning_rate=space["learning_rate"],
        n_estimators=space["n_estimators"],
        subsample=space["subsample"],
        colsample_bytree=space["colsample_bytree"],
        colsample_bylevel=space["colsample_bylevel"],
        colsample_bynode=space["colsample_bynode"],
        max_depth=space["max_depth"],
        min_child_weight=space["min_child_weight"],
        reg_alpha=space["reg_alpha"],
        reg_lambda=space["reg_lambda"],
        gamma=space["gamma"],
        eval_metric="error",
        early_stopping_rounds=100,
    )

    # Train the model on the training data and evaluate on the validation set
    model.fit(
        space["X_train"],
        space["y_train"],
        eval_set=[(space["X_valid"], space["y_valid"])],
        verbose=False,
    )

    # Predict on the validation data and calculate accuracy
    y_pred = model.predict(space["X_valid"])
    accuracy = accuracy_score(space["y_valid"], y_pred)

    # Return the negative accuracy as the loss to be minimized
    return {"loss": -round(accuracy, 3), "status": STATUS_OK}


"""
    This function returns the best hyper-parameters for the XGBoost
    from given training and the validation data using Hyperopt.

    Args:
    - X_train: contain data for training features.
    - y_train: contain data for training targets.
    - X_valid: containdata for validation features.
    - y_valid: contain data for validation target.
"""


def find_hyper_params(X_train, y_train, X_valid, y_valid):

    # Define the search space for hyperparameters
    space = {
        "base_score": hp.choice(
            "base_score", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        ),
        "scale_pos_weight": hp.choice(
            "scale_pos_weight", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        ),
        "learning_rate": hp.choice(
            "learning_rate",
            [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        ),
        "n_estimators": hp.choice("n_estimators", [100, 500, 1000, 1500, 2000, 5000]),
        "colsample_bytree": hp.choice(
            "colsample_bytree", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        ),
        "subsample": hp.choice(
            "subsample", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        ),
        "colsample_bylevel": hp.choice(
            "colsample_bylevel",
            [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        ),
        "colsample_bynode": hp.choice(
            "colsample_bynode", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        ),
        "max_depth": hp.choice(
            "max_depth", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40]
        ),
        "min_child_weight": hp.choice("min_child_weight", [0.01, 0.1, 1, 10, 100]),
        "reg_alpha": hp.choice(
            "reg_alpha",
            [0.000001, 0.000005, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        ),
        "reg_lambda": hp.choice(
            "reg_lambda",
            [0.000001, 0.000005, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        ),
        "gamma": hp.choice("gamma", [0, 0.01, 0.1, 1, 10, 100]),
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
    }

    # Use Hyperopt to find the best hyperparameters
    best_hyperparams = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=1000,
        trials=Trials(),
    )

    print(best_hyperparams)
    print("\n")


"""
    This function fits XGBoost on given training using relevent hyperparameters, 
    then evaluates model prior saving that to a file.


    Args:
    - X_train: contain data for training features.
    - y_train: contain data for training target.
    - X_valid: contain data for validation features
    - y_valid: contain data for validation target.
    - X_test: contain data for test features.
    - y_test: contain data for test target
    - hyper: hyperparameters dict
    - pattern: String pattern for using filename while model is saved.
"""


def fit_model(X_train, y_train, X_valid, y_valid, X_test, y_test, hyper, pattern):
    # Define the search space for hyperparameters
    space = {
        "base_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        "scale_pos_weight": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "learning_rate": [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        "n_estimators": [100, 500, 1000, 1500, 2000, 5000],
        "colsample_bytree": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "subsample": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "colsample_bylevel": [
            0.001,
            0.01,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1,
        ],
        "colsample_bynode": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40],
        "min_child_weight": [0.01, 0.1, 1, 10, 100],
        "reg_alpha": [
            0.000001,
            0.000005,
            0.00001,
            0.0001,
            0.001,
            0.01,
            0.1,
            1,
            10,
            100,
        ],
        "reg_lambda": [
            0.000001,
            0.000005,
            0.00001,
            0.0001,
            0.001,
            0.01,
            0.1,
            1,
            10,
            100,
        ],
        "gamma": [0, 0.01, 0.1, 1, 10, 100],
    }

    # Initialize the XGBoost classifier with selected hyperparameters
    model = xgb.XGBClassifier(
        n_jobs=-1,
        random_state=42,
        booster="gbtree",
        objective="binary:logistic",
        use_label_encoder=False,
        base_score=space["base_score"][hyper["base_score"]],
        scale_pos_weight=space["scale_pos_weight"][hyper["scale_pos_weight"]],
        learning_rate=space["learning_rate"][hyper["learning_rate"]],
        n_estimators=space["n_estimators"][hyper["n_estimators"]],
        subsample=space["subsample"][hyper["subsample"]],
        colsample_bytree=space["colsample_bytree"][hyper["colsample_bytree"]],
        colsample_bylevel=space["colsample_bylevel"][hyper["colsample_bylevel"]],
        colsample_bynode=space["colsample_bynode"][hyper["colsample_bynode"]],
        max_depth=space["max_depth"][hyper["max_depth"]],
        min_child_weight=space["min_child_weight"][hyper["min_child_weight"]],
        reg_alpha=space["reg_alpha"][hyper["reg_alpha"]],
        reg_lambda=space["reg_lambda"][hyper["reg_lambda"]],
        gamma=space["gamma"][hyper["gamma"]],
        eval_metric="error",
        early_stopping_rounds=100,
    )

    # Train the model on the training data and evaluate on the validation set
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    # Calculate and print statistics
    statistics(X_train, y_train, X_valid, y_valid, X_test, y_test, model)

    # Save the trained model to a file
    dump(
        model,
        "modelling/" + pattern + ".joblib",
    )


"""
    Compute and print different essential performance metrics.

    Args:
    - X_train: contain data for training features.
    - y_train: contain data for training target.
    - X_valid: contain data for validation features.
    - y_valid: contain data for validation target.
    - X_test: contain data for test features.
    - y_test: contain data for test target.
    - model: Trained model.
"""


def statistics(X_train, y_train, X_valid, y_valid, X_test, y_test, model):
    # Calculate training accuracy
    train_accuracy = model.score(X_train, y_train)

    # Predict and calculate validation accuracy
    y_pred = model.predict(X_valid)
    valid_accuracy = accuracy_score(y_valid, y_pred)

    # Predict and calculate test accuracy
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    # Compute confusion matrix and classification report for test data
    matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, labels=[0, 1], zero_division=1)

    # Print performance metrics
    print("Training Accuracy: %.3f" % train_accuracy)
    print("Valid Accuracy: %.3f" % valid_accuracy)
    print("Testing Accuracy: %.3f" % test_accuracy)
    print("Confusion Matrix:")
    print(matrix)
    print("Classification Statistics:")
    print(report)
