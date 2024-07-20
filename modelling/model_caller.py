import numpy as np

from features.extractor import Extractor


# Predict using the model
def predict(clf, start, reversal1, msb, reversal2, end, logging):
    """
    Uses features to make a prediction with the classifier.

    Args:
        clf (Classifier): A trained classifier to make the predictions.
        start (float): Observation start time.
        reversal1 (float): Observation first reversal time.
        msb (float): Observation MSB value.
        reversal2 (float): Observation second reversal time.
        end (float): Observation end time.
        logging (bool): Logging enabled via flagging.

    Returns:
        int: Result predicted (as 1 or 0).

    """
    if not logging or clf is None:
        return 1

    observation = Extractor()._extract_features(
        "", start, reversal1, msb, reversal2, end, 0
    )

    for key in observation:
        if isinstance(observation[key], list) and all(
            isinstance(i, bool) for i in observation[key]
        ):
            observation[key] = [int(i) for i in observation[key]]

    first_key = next(iter(observation))
    observation.pop(first_key)

    pair = np.array(list(observation.values())).T
    pair = pair.reshape(-1, 1)

    pair = pair.reshape(1, -1)
    result = clf.predict(pair)

    return result[0]
