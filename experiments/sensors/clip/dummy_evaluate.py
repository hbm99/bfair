from itertools import combinations
import json
import random
from bfair.datasets.fairface import GENDER_VALUES, RACE_VALUES
from bfair.datasets.noisymultifairface import load_dataset as load_noisymultifairface
from bfair.sensors.base import P_GENDER, P_RACE
from bfair.sensors.optimization import compute_errors, compute_scores


def _get_scores(values, y, y_pred):
    errors = compute_errors(y, y_pred, values)
    print(errors)

    scores = compute_scores(errors)
    print(scores)
    return scores


def _get_list(y):
    new_y = [None for _ in range(len(y))]
    for i in range(len(y)):
        if isinstance(y.values[i], str):
            new_y[i] = [y.values[i]]
        else:
            new_y[i] = y.values[i]
    return new_y


def _get_mock_model_predictions(y_size, mock_model):
    y_pred = [[] for _ in range(y_size)]
    for i in range(len(y_pred)):
        if isinstance(mock_model, str):
            y_pred[i].append(mock_model)
        else:
            y_pred[i].extend(mock_model)
    return y_pred


def _get_random_uniform_predictions(y_size, representations):
    random.seed(0)
    y_pred = [None for _ in range(y_size)]
    for i in range(len(y_pred)):
        y_pred[i] = representations[random.randint(0, len(representations) - 1)]
    return y_pred


def get_baseline_score(
    dataset, attribute, attribute_values, baseline="random-uniform", mock_model=None
):
    y = dataset.data[attribute]
    y = _get_list(y)
    if baseline == "mock_model":
        assert mock_model is not None
        y_pred = _get_mock_model_predictions(len(y), mock_model)
    elif baseline == "random-uniform":
        representations = [
            list(combinations(attribute_values, i))
            for i in range(1, len(attribute_values) + 1)
        ]
        representations = [item for sublist in representations for item in sublist]
        representations.append([])
        y_pred = _get_random_uniform_predictions(len(y), representations)

    scores = _get_scores(attribute_values, y, y_pred)
    return scores


if __name__ == "__main__":
    dataset = load_noisymultifairface(split_seed=0, balanced=True)

    json_results = {}

    attribute = P_GENDER
    attribute_values = GENDER_VALUES

    baseline = "mock_model"

    scores_always_none = get_baseline_score(
        dataset, attribute, attribute_values, baseline=baseline, mock_model=""
    )
    scores_always_both = get_baseline_score(
        dataset,
        attribute,
        attribute_values,
        baseline=baseline,
        mock_model=["Male", "Female"],
    )
    scores_always_male = get_baseline_score(
        dataset, attribute, attribute_values, baseline=baseline, mock_model="Male"
    )
    scores_always_female = get_baseline_score(
        dataset, attribute, attribute_values, baseline=baseline, mock_model="Female"
    )

    gender_mocks = {
        "always-none": scores_always_none,
        "always-male": scores_always_male,
        "always-female": scores_always_female,
        "always-both": scores_always_both,
    }

    json_results["gender-mocks"] = gender_mocks

    baseline = "random-uniform"

    gender_scores_random_uniform = get_baseline_score(
        dataset, attribute, attribute_values
    )

    json_results["gender_random_uniform"] = gender_scores_random_uniform

    attribute = P_RACE
    attribute_values = RACE_VALUES

    race_representations = [
        item
        for sublist in [
            list(combinations(attribute_values, i))
            for i in range(1, len(attribute_values) + 1)
        ]
        for item in sublist
    ]

    race_mocks = {
        "always-"
        + "-".join(repr): get_baseline_score(
            dataset, attribute, attribute_values, baseline="mock_model", mock_model=repr
        )
        for repr in race_representations
    }
    race_mocks["always-none"] = get_baseline_score(
        dataset, attribute, attribute_values, baseline="mock_model", mock_model=""
    )

    json_results["race-mocks"] = race_mocks

    race_scores_random_uniform = get_baseline_score(
        dataset, attribute, attribute_values
    )

    json_results["race_random_uniform"] = race_scores_random_uniform

    with open("results/dummy_evaluate/baseline_scores.json", "a") as f:
        f.write(json.dumps(json_results, indent=4))
