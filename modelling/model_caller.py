import numpy as np

from features.extractor import Extractor


# Predict using the model
def predict(clf, start, reversal1, msb, reversal2, end, logging):
    if not logging:
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
