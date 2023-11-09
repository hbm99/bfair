from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from bfair.datasets.fairface import GENDER_VALUES, IMAGE_COLUMN
from bfair.datasets.noisymultifairface import load_dataset as load_noisymultifairface
from bfair.sensors.base import P_GENDER, P_RACE
from bfair.sensors.handler import ImageSensorHandler
from bfair.sensors.image.clip.base import ClipBasedSensor
from bfair.sensors.image.clip.finetuned_clip.base import FinetunedClipSensor
from bfair.sensors.optimization import compute_errors, compute_scores
from autogoal.kb import Matrix
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

target = P_GENDER
target_values = GENDER_VALUES
dataset = load_noisymultifairface(split_seed=0, balanced=True)
sensor_type = FINETUNED_SENSOR
filtering_pipeline = None
learner = MODELS["logistic_regression"]
logits_to_probs = "sigmoid"
tokens_pipeline = [[value.lower() for value in target_values]]
target_column = None


X_train = dataset.data[IMAGE_COLUMN]
y_train = dataset.data[target]
X_test = dataset.test[IMAGE_COLUMN]
y_test = dataset.test[target]


sensor = sensor_type.build(
    filtering_pipeline, learner, logits_to_probs, tokens_pipeline
)
handler = ImageSensorHandler([sensor])

handler.fit(X_train, y_train, target_values, target)
y_pred = handler.annotate(X_test, Matrix, target_values, target)
# score = score_func(y_test, y_pred)

# y_pred = handler(X_test, target_values, target)
errors = compute_errors(y_test, y_pred, target_values)
print(errors)

scores = compute_scores(errors)
print(scores)

if target_column is not None:
    fairness = exploded_statistical_parity(
        data=dataset.data,
        protected_attributes=P_GENDER,
        target_attribute=target_column,
        target_predictions=None,
        positive_target="positive",
        return_probs=True,
    )
    print(dataset.data)
    print("PA: gender. True fairness:", fairness)

    auto_annotated = dataset.data.copy()
    auto_annotated[P_GENDER] = [list(x) for x in y_pred]

    fairness = exploded_statistical_parity(
        data=auto_annotated,
        protected_attributes=P_GENDER,
        target_attribute=target_column,
        target_predictions=None,
        positive_target="positive",
        return_probs=True,
    )
    print(auto_annotated)
    print("PA: gender. Estimated fairness:", fairness)

    fairness = exploded_statistical_parity(
        data=dataset.data,
        protected_attributes=P_RACE,
        target_attribute=target_column,
        target_predictions=None,
        positive_target="positive",
        return_probs=True,
    )
    print(dataset.data)
    print("PA: race. True fairness:", fairness)

    auto_annotated = dataset.data.copy()
    auto_annotated[P_RACE] = [list(x) for x in y_pred]

    fairness = exploded_statistical_parity(
        data=auto_annotated,
        protected_attributes=P_RACE,
        target_attribute=target_column,
        target_predictions=None,
        positive_target="positive",
        return_probs=True,
    )
    print(auto_annotated)
    print("PA: race. Estimated fairness:", fairness)
