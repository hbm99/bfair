
import pandas as pd

from bfair.datasets import load_utkface
from bfair.datasets.utkface import GENDER_VALUES, RACE_VALUES
# from bfair.metrics import exploded_accuracy_disparity, exploded_statistical_parity, exploded_equalized_odds, exploded_equal_opportunity
from bfair.sensors.base import P_GENDER, P_RACE
from bfair.sensors.image.clip.base import ClipBasedSensor
from bfair.sensors.optimization import compute_errors, compute_scores
from bfair.sensors.text.embedding.filters import BestScoreFilter


def main():
    
    clip_sensor = ClipBasedSensor(BestScoreFilter())
    print("Loaded!")

    dataset = load_utkface(split_seed=0)
    
    
    X = dataset.data['image']
    y = dataset.data['race']

    predictions = clip_sensor(X, RACE_VALUES, P_RACE)

    new_y = [[] for _ in range(len(y))]
    for i in range(len(y)):
        new_y[i].append(y.values[i])

    new_predictions = [[] for _ in range(len(predictions))]
    for i in range(len(predictions)):
        pred_i = predictions[i][1]
        for j in range(len(pred_i)):
            new_predictions[i].append(pred_i[j][0])


    errors = compute_errors(new_y, new_predictions, RACE_VALUES)
    print(errors)

    scores = compute_scores(errors)
    print(scores)

    # results = pd.concat(
    #     (
    #         X,
    #         y.str.join(" & "),
    #         pd.Series(predictions, name="Predicted", index=X.index).str.join(" & "),
    #     ),
    #     axis=1,
    # )
    # print(results)

    # fairness = exploded_accuracy_disparity(
    #     data=dataset.data,
    #     protected_attributes= ['gender', 'race'],
    #     target_attribute='gender',
    #     target_predictions=None,
    #     positive_target='0', # '0' = male, '1' = female
    #     return_probs=True,
    # )
    # print(dataset.data)
    # print("True fairness:", fairness)

    # auto_annotated = dataset.data.copy()
    # auto_annotated['gender'] = [list(x) for x in predictions]

    # fairness = exploded_accuracy_disparity(
    #     data=dataset.data,
    #     protected_attributes= ['gender', 'race'],
    #     target_attribute='gender',
    #     target_predictions=None,
    #     positive_target='0', # '0' = male, '1' = female
    #     return_probs=True,
    # )
    # print(auto_annotated)
    # print("Estimated fairness:", fairness)


if __name__ == "__main__":
    main()
