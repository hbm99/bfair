
import pandas as pd

from bfair.datasets import load_utkface
from bfair.datasets.utkface import GENDER_VALUES
from bfair.metrics import exploded_statistical_parity
from bfair.sensors.base import P_GENDER
from bfair.sensors.handler import SensorHandler
from bfair.sensors.image.clip.base import ClipBasedSensor
from bfair.sensors.optimization import compute_errors, compute_scores


def main():
    
    clip_sensor = ClipBasedSensor()
    print("Loaded!")

    dataset = load_utkface(split_seed=0)
    
    
    X = dataset.data['image']
    y = dataset.data['gender']

    predictions = clip_sensor(X, GENDER_VALUES, P_GENDER)

    errors = compute_errors(y, predictions, GENDER_VALUES)
    print(errors)

    scores = compute_scores(errors)
    print(scores)

    results = pd.concat(
        (
            X,
            y.str.join(" & "),
            pd.Series(predictions, name="Predicted", index=X.index).str.join(" & "),
        ),
        axis=1,
    )
    print(results)

    fairness = exploded_statistical_parity(
        data=dataset.data,
        protected_attributes='gender',
        target_attribute='race',
        target_predictions=None,
        positive_target='0', # '0' = white, '1' = black, '2' = asian, '3' = indian, '4' = others
        return_probs=True,
    )
    print(dataset.data)
    print("True fairness:", fairness)

    auto_annotated = dataset.data.copy()
    auto_annotated['gender'] = [list(x) for x in predictions]

    fairness = exploded_statistical_parity(
        data=auto_annotated,
        protected_attributes='gender',
        target_attribute='race',
        target_predictions=None,
        positive_target="0",
        return_probs=True,
    )
    print(auto_annotated)
    print("Estimated fairness:", fairness)


if __name__ == "__main__":
    main()
