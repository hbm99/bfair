
import pandas as pd

from bfair.datasets import load_utkface
from bfair.datasets.utkface import GENDER_VALUES, RACE_VALUES
from bfair.metrics import exploded_accuracy_disparity
from bfair.sensors.base import P_GENDER, P_RACE
from bfair.sensors.image.clip.base import ClipBasedSensor
from bfair.sensors.optimization import compute_errors, compute_scores
from bfair.sensors.text.embedding.filters import BestScoreFilter

POSITIVE_TARGETS = {P_GENDER: 'Male', P_RACE: 'White'}


def evaluate(values, attribute, phrases=None):

    if not phrases:
        phrases = [attribute + ': ' + attr for attr in values]
    
    clip_sensor = ClipBasedSensor(BestScoreFilter())
    print("Loaded!")

    dataset = load_utkface(split_seed=0)
    
    
    X = dataset.data['image']
    y = dataset.data[attribute]

    predictions = clip_sensor(X, values, phrases)

    new_y = [[] for _ in range(len(y))]
    for i in range(len(y)):
        new_y[i].append(y.values[i])

    new_predictions = [[] for _ in range(len(predictions))]
    for i in range(len(predictions)):
        pred_i = predictions[i][1]
        for j in range(len(pred_i)):
            new_predictions[i].append(pred_i[j][0])


    errors = compute_errors(new_y, new_predictions, values)
    print(errors)

    scores = compute_scores(errors)
    print(scores)

    fairness = exploded_accuracy_disparity(
        data=dataset.data,
        protected_attributes= [P_GENDER, P_RACE],
        target_attribute=attribute,
        target_predictions= [new_predictions[i][0] for i in range(len(new_predictions))],
        positive_target=POSITIVE_TARGETS[attribute],
        return_probs=True,
    )
    print(dataset.data)
    print("Fairness:", fairness)


if __name__ == "__main__":
    # phrases = [P_RACE + ': ' + attr for attr in RACE_VALUES]
    # phrases = ['This is a person of ' + attr + ' ' + P_RACE for attr in RACE_VALUES]
    evaluate(RACE_VALUES, P_RACE)
