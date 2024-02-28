import os
import pickle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from autogoal.kb import Matrix
from bfair.datasets.fairface import GENDER_VALUES, IMAGE_COLUMN, RACE_VALUES
from bfair.datasets.noisymultifairface import load_dataset as load_noisymultifairface
from bfair.sensors.base import P_GENDER, P_RACE
from bfair.sensors.handler import ImageSensorHandler
from bfair.sensors.image.vl.clip.base import ClipBasedSensor
from bfair.sensors.image.vl.clip.finetuned_clip.base import FinetunedClipSensor
from bfair.sensors.image.vl.clip.optimization import eval_preprocess
from bfair.sensors.optimization import compute_errors, compute_scores
from bfair.metrics import exploded_statistical_parity

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

metric = "test_acc"
attribute = P_GENDER
attribute_values = GENDER_VALUES
dataset = load_noisymultifairface(split_seed=0, balanced=True, decision_columns=True)
sensor_type = FINETUNED_SENSOR
filtering_pipeline = None
learner = MODELS["random_forest"]
logits_to_probs = "sigmoid"
tokens_pipeline = [
    ["A person of " + attribute + " " + value.lower() for value in attribute_values]
]
target_column = attribute + "_biased_decision"

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
    sensor = sensor_type(
        model = None,
        filtering_pipeline=filtering_pipeline, # type: ignore
        learner=learner,
        tokens_pipeline=tokens_pipeline,
        logits_to_probs=logits_to_probs,
    )

    handler = ImageSensorHandler([sensor])
    handler.fit(X_train, y_train, attribute_values, attribute)
    y_pred = handler.annotate(X_test, Matrix, attribute_values, attribute)
    with open(file_name, "wb") as fp:
        pickle.dump(y_pred, fp)

y_test, y_pred, attributes = eval_preprocess(y_test, y_pred, attribute_values)

errors = compute_errors(y_test, y_pred, attributes)
print(errors)

scores = compute_scores(errors)
print(scores)

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
