import numpy as np

from features.extractor import Extractor


"""
    Uses features to make a prediction with the classifier.

    Args:
        clf (Classifier): A trained classifier to make the predictions.
        start (float): Observation start candle.
        reversal1 (float): Observation first reversal candle.
        msb (float): Observation MSB candle.
        reversal2 (float): Observation second reversal candle.
        end (float): Observation end candle.
        logging (bool): Logging enabled via flagging.

    Returns:
        int: Result predicted (as 1 or 0).

"""


def predict(clf, start, reversal1, msb, reversal2, end, logging):
    # Return 1 if logging is disabled or if no classifier is provided
    if not logging or clf is None:
        return 1

    # Extract features from the given candlestick observation
    observation = Extractor()._extract_features(
        "", start, reversal1, msb, reversal2, end, 0
    )

    # Convert any boolean lists within the observation to integers
    for key in observation:
        if isinstance(observation[key], list) and all(
            isinstance(i, bool) for i in observation[key]
        ):
            observation[key] = [int(i) for i in observation[key]]

    # Remove the first key-value pair from the observation
    first_key = next(iter(observation))
    observation.pop(first_key)

    # Convert the observation dictionary values into a NumPy array and ensure compatibility for prediction
    pair = np.array(list(observation.values())).T
    pair = pair.reshape(-1, 1)
    pair = pair.reshape(1, -1)
    result = clf.predict(pair)

    # Return the predicted result (1 or 0)
    return result[0]
