from functools import partial

from autogoal.kb import Matrix
from autogoal.sampling import Sampler
from autogoal.search import ConsoleLogger, NSPESearch
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


from bfair.methods.autogoal.ensembling.sampling import LogSampler, SampleModel
from bfair.sensors.handler import ImageSensorHandler
from bfair.sensors.image.clip.base import ClipBasedSensor
from bfair.sensors.image.clip.finetuned_clip.base import FinetunedClipSensor
from bfair.sensors.optimization import MACRO_F1, compute_errors, compute_scores
from bfair.sensors.text.embedding.filters import (
    BestScoreFilter,
    IdentityFilter,
    LargeEnoughFilter,
)

BATCH_SIZE = 64

MODELS = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "gradient_boosting_classifier": GradientBoostingClassifier,
    "svm": SVC,
    "knn": KNeighborsClassifier,
}


def optimize(
    X_train,
    y_train,
    X_test,
    y_test,
    attributes,
    attr_cls,
    score_key=MACRO_F1,
    consider_clip_based_sensor=True,
    force_clip_based_sensor=False,
    *,
    pop_size,
    search_iterations,
    evaluation_timeout,
    memory_limit,
    search_timeout,
    errors="warn",
    log_path=None,
    inspect=False,
    output_stream=None,
):
    score_key = score_key if isinstance(score_key, (list, tuple)) else [score_key]

    loggers = get_loggers(
        log_path=log_path,
    )

    search = NSPESearch(
        generator_fn=partial(
            generate,
            consider_clip_based_sensor=consider_clip_based_sensor,
            force_clip_based_sensor=force_clip_based_sensor,
            attr_cls=attr_cls,
            attributes=attributes,
        ),
        fitness_fn=build_fn(
            X_train,
            y_train,
            X_test,
            y_test,
            Matrix,
            attributes,
            attr_cls,
            score_func=build_score_fn(attributes, score_key),
        ),
        maximize=[True] * len(score_key),
        pop_size=pop_size,
        evaluation_timeout=evaluation_timeout,
        memory_limit=memory_limit,
        search_timeout=search_timeout,
        errors=errors,
    )
    best_solution, best_fn = search.run(generations=search_iterations, logger=loggers)

    if inspect:
        y_pred, counter, scores = evaluate(
            best_solution, X_train, y_train, attributes, attr_cls, Matrix
        )
        print("Results @ Training ...", file=output_stream)
        print(counter, file=output_stream)
        print(scores, file=output_stream)

        y_pred, counter, scores = evaluate(
            best_solution, X_test, y_test, attributes, attr_cls, Matrix
        )
        print("Results @ Testing ....", file=output_stream)
        print(counter, file=output_stream)
        print(scores, file=output_stream)

    return best_solution, best_fn, search


def get_loggers(log_path=None):
    loggers = [ConsoleLogger()]

    if log_path:
        from bfair.utils.autogoal import FileLogger

        file_logger = FileLogger(output_path=log_path)
        loggers.append(file_logger)

    return loggers


def generate(
    sampler: Sampler,
    consider_clip_based_sensor=True,
    force_clip_based_sensor=False,
    *,
    attr_cls,
    attributes,
):
    """
    Generates a new SampleModel object with the given Sampler.
    """
    sampler = LogSampler(sampler)
    sensors = []
    if force_clip_based_sensor or (
        consider_clip_based_sensor and sampler.boolean("include-clip-sensor")
    ):
        sensor = get_clip_based_sensor(sampler, attr_cls, attributes)
        sensors.append(sensor)

    handler = ImageSensorHandler(sensors, merge=None)
    return SampleModel(sampler, handler)


def get_clip_based_sensor(sampler: LogSampler, attr_cls, attributes):
    prefix = "clip-sensor."

    tokens_pipeline = get_tokens_pipeline(sampler, attr_cls, attributes, prefix)

    selection = sampler.choice(["filter", "learner"], handle=f"{prefix}selection")

    filtering_pipeline = None
    learner = None
    if selection == "filter":
        filtering_pipeline = get_filtering_pipeline(sampler, prefix)
    elif selection == "learner":
        learner = get_learning_pipeline(sampler, prefix)

    sensor = ClipBasedSensor.build(
        filtering_pipeline=filtering_pipeline,
        learner=learner,
        tokens_pipeline=tokens_pipeline,
    )
    return sensor


def get_tokens_pipeline(sampler: LogSampler, attr, attr_values, prefix):
    phrase_sentences = get_phrase(sampler, attr, attr_values, prefix=f"{prefix}phrase")
    return [phrase_sentences]


def get_phrase(sampler: LogSampler, attr, attr_values, prefix):
    options = {
        "__attr__": [value for value in attr_values],
        attr + ": __attr__": [attr + ": " + value for value in attr_values],
        "This is a person of __attr__ "
        + attr: ["This is a person of " + value + " " + attr for value in attr_values],
        "This is a person of "
        + attr
        + " __attr__": [
            "This is a person of " + attr + " " + value for value in attr_values
        ],
        "A person of __attr__ "
        + attr: ["A person of " + value + " " + attr for value in attr_values],
        "A person of "
        + attr
        + " __attr__": ["A person of " + attr + " " + value for value in attr_values],
        "A __attr__ "
        + attr
        + " person": ["A " + value + " " + attr + " person" for value in attr_values],
        "An image of a person of __attr__ "
        + attr: [
            "An image of a person of " + value + " " + attr for value in attr_values
        ],
    }
    phrase = sampler.choice(list(options.keys()), handle=f"{prefix}")
    return options[phrase]


def get_learning_pipeline(sampler: LogSampler, prefix):
    model_name = sampler.choice(list(MODELS.keys()), handle=f"{prefix}model")
    return MODELS[model_name]


def get_filtering_pipeline(sampler: LogSampler, prefix):
    filtering_pipeline = []

    filter = get_filter(sampler, allow_none=True, prefix=f"{prefix}filter")
    filtering_pipeline.append(filter)

    return filtering_pipeline


def get_filter(sampler: LogSampler, allow_none: bool, prefix: str):
    options = ["LargeEnoughFilter", "BestScoreFilter"]
    if allow_none:
        options.append("None")

    filter_name = sampler.choice(options, handle=f"{prefix}-filter")

    if filter_name == "LargeEnoughFilter":
        norm_threshold = sampler.continuous(
            0, 1, handle=f"{prefix}-large-norm-threshold"
        )
        return LargeEnoughFilter(norm_threshold)

    elif filter_name == "BestScoreFilter":
        relative_threshold = sampler.continuous(
            0, 1, handle=f"{prefix}-best-relative-threshold"
        )
        norm_threshold = sampler.continuous(
            0, 1, handle=f"{prefix}-best-norm-threshold"
        )
        return BestScoreFilter(
            threshold=relative_threshold,
            zero_threshold=norm_threshold,
        )

    elif filter_name == "None" and allow_none:
        return IdentityFilter()

    else:
        raise ValueError(filter_name)


def fn(
    generated: SampleModel,
    X_train,
    y_train,
    X_test,
    y_test,
    stype,
    attributes,
    attr_cls,
    score_func,
):
    handler: ImageSensorHandler = generated.model
    handler.fit(X_train, y_train, attributes, attr_cls)
    y_pred = handler.annotate(X_test, stype, attributes, attr_cls)
    score = score_func(y_test, y_pred)
    return score


def build_fn(X_train, y_train, X_test, y_test, stype, attributes, attr_cls, score_func):
    return partial(
        fn,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        stype=stype,
        attributes=attributes,
        attr_cls=attr_cls,
        score_func=score_func,
    )


def score_fn(X, y, attributes, score_keys):
    X, y, attributes = eval_preprocess(X, y, attributes)
    errors = compute_errors(X, y, attributes)
    scores = compute_scores(errors)
    return tuple(scores[key] for key in score_keys)


def eval_preprocess(X, y, attributes):
    new_X = []
    for x in X:
        if isinstance(x, str) and x in attributes:
            new_X.append([x.lower()])
        elif isinstance(x, str) and x not in attributes:
            new_X.append([])
        else:
            new_X.append([s.lower() for s in x])
    X = new_X
    y = [[s.lower() for s in lst] for lst in y]
    attributes = [attr.lower() for attr in attributes]
    return X, y, attributes


def build_score_fn(attributes, score_keys):
    return partial(score_fn, attributes=attributes, score_keys=score_keys)


def evaluate(solution, X, y, attributes, attr_cls, autogoal_type):
    handler: ImageSensorHandler = solution.model
    y_pred = handler.annotate(X, autogoal_type, attributes, attr_cls)
    y, y_pred, attributes = eval_preprocess(y, y_pred, attributes)
    errors = compute_errors(y, y_pred, attributes)
    scores = compute_scores(errors)
    return y_pred, errors, scores
