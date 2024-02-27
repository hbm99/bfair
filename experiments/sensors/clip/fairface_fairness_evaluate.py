# from itertools import combinations
# from statistics import mean
import os
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from bfair.datasets.fairface import GENDER_VALUES, IMAGE_COLUMN, RACE_VALUES
from bfair.datasets.noisymultifairface import load_dataset as load_noisymultifairface
from bfair.sensors.base import P_GENDER, P_RACE
from bfair.sensors.handler import ImageSensorHandler
from bfair.sensors.image.clip.base import ClipBasedSensor
from bfair.sensors.image.clip.finetuned_clip.base import FinetunedClipSensor
from bfair.sensors.image.clip.optimization import eval_preprocess
from bfair.sensors.optimization import compute_errors, compute_scores
from autogoal.kb import Matrix
from bfair.metrics import exploded_statistical_parity

# from experiments.sensors.clip.dummy_evaluate import _get_random_uniform_predictions
# from statistics import stdev
import pickle

MODELS = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "gradient_boosting_classifier": GradientBoostingClassifier,
    "svm": SVC,
    "knn": KNeighborsClassifier,
}

BASED_SENSOR = ClipBasedSensor
FINETUNED_SENSOR = FinetunedClipSensor

### CONFIGURATION ###

metric = "f1"
attribute = P_RACE
attribute_values = RACE_VALUES
dataset = load_noisymultifairface(split_seed=0, balanced=True, decision_columns=True)
sensor_type = BASED_SENSOR
filtering_pipeline = None
learner = MODELS["svm"]
logits_to_probs = "softmax"
tokens_pipeline = [
    [
        "An image of a person of " + value.lower() + " " + attribute
        for value in attribute_values
    ]
]
target_column = "race_biased_decision"

### END CONFIGURATION ###


# Convert attribute values to lists, i.e., from: "male", "female", "", ["male", "female"] ----> ["male"], ["female"], [], ["male", "female"]
converter = lambda x: [] if x == "" else [x] if isinstance(x, str) else x

dataset.test[attribute] = dataset.test[attribute].apply(converter)


X_train = dataset.data[IMAGE_COLUMN]
y_train = dataset.data[attribute]
X_test = dataset.test[IMAGE_COLUMN]
y_test = dataset.test[attribute]


file_name = attribute + "_best_" + metric + "_predictions"
if os.path.isfile(os.path.join(".", file_name)):
    with open(file_name, "rb") as fp:
        y_pred = pickle.load(fp)
else:
    sensor = sensor_type.build(
        filtering_pipeline, learner, logits_to_probs, tokens_pipeline
    )
    handler = ImageSensorHandler([sensor])
    handler.fit(X_train, y_train, attribute_values, attribute)
    y_pred = handler.annotate(X_test, Matrix, attribute_values, attribute)
    with open(file_name, "wb") as fp:
        pickle.dump(y_pred, fp)

# for classif in attribute_values:
#     y_pred = [{classif} for _ in range(len(y_test))]

# y_pred = y_test.copy()

y_test, y_pred, attributes = eval_preprocess(y_test, y_pred, attribute_values)

errors = compute_errors(y_test, y_pred, attributes)
print(errors)

scores = compute_scores(errors)
print(scores)

# scenario 2.1
if target_column is not None:
    fairness = exploded_statistical_parity(
        data=dataset.test,
        protected_attributes=attribute,
        target_attribute=target_column,
        target_predictions=None,
        positive_target=1,
        return_probs=True,
    )
    print(dataset.test)
    print(f"PA: {attribute}. True fairness:", fairness)

    auto_annotated = dataset.test.copy()
    auto_annotated[attribute] = [list(x) for x in y_pred]

    fairness = exploded_statistical_parity(
        data=auto_annotated,
        protected_attributes=attribute,
        target_attribute=target_column,
        target_predictions=None,
        positive_target=1,
        return_probs=True,
    )
    print(auto_annotated)
    print(f"PA: {attribute}. Estimated fairness:", fairness)

# # random-uniform
# representations = [
#     list(combinations(attribute_values, i)) for i in range(1, len(attribute_values) + 1)
# ]
# representations = [item for sublist in representations for item in sublist]
# representations.append([])
# y_pred = _get_random_uniform_predictions(len(y_test), representations)

# y_test, y_pred, attributes = eval_preprocess(y_test, y_pred, attribute_values)
# errors = compute_errors(y_test, y_pred, attributes)
# print(errors)

# scores = compute_scores(errors)
# print(scores)

# auto_annotated = dataset.test.copy()
# auto_annotated[attribute] = [list(x) for x in y_pred]

# fairness = exploded_statistical_parity(
#     data=auto_annotated,
#     protected_attributes=attribute,
#     target_attribute=target_column,
#     target_predictions=None,
#     positive_target=1,
#     return_probs=True,
# )
# print(auto_annotated)
# print(f"PA: {attribute}. random-uniform. Estimated fairness:", fairness)

# # mocks
# for race in RACE_VALUES:
#     y_pred = [{race} for _ in range(len(y_test))]

#     y_test, y_pred, attributes = eval_preprocess(y_test, y_pred, attribute_values)
#     errors = compute_errors(y_test, y_pred, attributes)
#     print(errors)

#     scores = compute_scores(errors)
#     print(scores)

#     auto_annotated = dataset.test.copy()
#     auto_annotated[attribute] = [list(x) for x in y_pred]

#     fairness = exploded_statistical_parity(
#         data=auto_annotated,
#         protected_attributes=attribute,
#         target_attribute=target_column,
#         target_predictions=None,
#         positive_target=1,
#         return_probs=True,
#     )
#     print(auto_annotated)
#     print(f"PA: {attribute}. always-" + race + ". Estimated fairness:", fairness)

# # scenario 2.2

# auto_annotated = dataset.test.copy()
# auto_annotated[attribute] = [list(x) for x in y_pred]

# mses = []
# deltas = []
# senses = []
# for i in range(30):
#     target_column = "decision_" + str(i)

#     # print()
#     # print(target_column)

#     true_fairness = exploded_statistical_parity(
#         data=dataset.test,
#         protected_attributes=attribute,
#         target_attribute=target_column,
#         target_predictions=None,
#         positive_target=1,
#         return_probs=True,
#     )
#     # print(dataset.test)
#     # print(f"PA: {attribute}. True fairness:", true_fairness)

#     fairness = exploded_statistical_parity(
#         data=auto_annotated,
#         protected_attributes=attribute,
#         target_attribute=target_column,
#         target_predictions=None,
#         positive_target=1,
#         return_probs=True,
#     )
#     # fairness = (
#     #     0,
#     #     {k: v for k, v in zip(attribute_values, [0 for _ in attribute_values])},
#     # )
#     # print(auto_annotated)
#     # print(f"PA: {attribute}. Estimated fairness:", fairness)

#     model_fairness, model_prs = fairness
#     gold_fairness, gold_prs = true_fairness

#     model_prs = {k.lower(): v for k, v in model_prs.items()}
#     gold_prs = {k.lower(): v for k, v in gold_prs.items()}

#     # MSE
#     model_vs_gold_pr_diff = []
#     for key, value in model_prs.items():
#         model_vs_gold_pr_diff.append(abs(value - gold_prs[key]))
#     mse = 0
#     for item in model_vs_gold_pr_diff:
#         mse += item**2
#     mse = mse / 2

#     mses.append(mse)

#     # print(target_column + " MSE: " + str(mse))

#     # delta 1 %
#     delta = abs(model_fairness - gold_fairness) // 0.01
#     deltas.append(delta)

#     # print(target_column + " delta 1%: " + str(delta))

#     # sense
#     max_model_prs = [
#         kv[0] for kv in model_prs.items() if kv[1] == max(model_prs.values())
#     ]
#     min_model_prs = [
#         kv[0] for kv in model_prs.items() if kv[1] == min(model_prs.values())
#     ]

#     max_gold_prs = [kv[0] for kv in gold_prs.items() if kv[1] == max(gold_prs.values())]
#     min_gold_prs = [kv[0] for kv in gold_prs.items() if kv[1] == min(gold_prs.values())]

#     sense = 0
#     if (
#         max_model_prs == max_gold_prs
#         and min_model_prs == min_gold_prs
#         and max_model_prs != min_model_prs
#     ):
#         sense = 1
#     senses.append(sense)

#     # print(target_column + " sense: " + str(sense))


# mse_mean = mean(mses)
# mse_stdv = stdev(mses)
# print("MSE (mean): " + str(mse_mean))
# print("MSE (stdv): " + str(mse_stdv))

# delta_mean = mean(deltas)
# delta_stdv = stdev(deltas)
# print("Delta 1% (mean): " + str(delta_mean))
# print("Delta 1% (stdv): " + str(delta_stdv))

# sense_mean = mean(senses)
# sense_stdv = stdev(senses)
# print("Sense (mean): " + str(sense_mean))
# print("Sense (stdv): " + str(sense_stdv))
