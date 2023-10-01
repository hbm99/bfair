from functools import partial
from statistics import mean
from typing import List

import clip
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from autogoal.kb import Matrix
from autogoal.sampling import Sampler
from autogoal.search import ConsoleLogger, NSPESearch
from PIL import Image

from bfair.methods.autogoal.ensembling.sampling import LogSampler, SampleModel
from bfair.sensors.handler import ImageSensorHandler
from bfair.sensors.image.clip.base import ClipBasedSensor
from bfair.sensors.optimization import MACRO_F1, compute_errors, compute_scores
from bfair.sensors.text.embedding.filters import (
    BestScoreFilter,
    IdentityFilter,
    LargeEnoughFilter,
    NonEmptyFilter,
)

BATCH_SIZE = 64


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
            X_train=X_train,
            y_train=y_train,
        ),
        fitness_fn=build_fn(
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
    X_train,
    y_train,
):
    """
    Generates a new SampleModel object with the given Sampler.
    """
    sampler = LogSampler(sampler)
    sensors = []
    if force_clip_based_sensor or (
        consider_clip_based_sensor and sampler.boolean("include-clip-sensor")
    ):
        sensor = get_clip_based_sensor(sampler, attr_cls, attributes, X_train, y_train)
        sensors.append(sensor)

    handler = ImageSensorHandler(sensors, merge=None)
    return SampleModel(sampler, handler)


def get_clip_based_sensor(sampler: LogSampler, attr_cls, attributes, X_train, y_train):
    prefix = "clip-sensor."

    tokens_pipeline = get_tokens_pipeline(sampler, attr_cls, attributes, prefix)

    selection = sampler.choice(["filter", "learner"], handle=f"{prefix}selection")

    filtering_pipeline = None
    learner = None
    if selection == "filter":
        filtering_pipeline = get_filtering_pipeline(sampler, prefix)
    elif selection == "learner":
        learner = get_learning_pipeline(
            sampler,
            prefix,
            tokens_pipeline,
            X_train,
            y_train,
            attributes=attributes,
            attr_cls=attr_cls,
        )

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
    phrase = sampler.choice(list(options.keys()), handle=f"{prefix}-phrase")
    return options[phrase]


def get_learning_pipeline(
    sampler: LogSampler, prefix, tokens_pipeline, X_train, y_train, attributes, attr_cls
):
    image_list = X_train

    # get labels for each image
    image_labels = clip_sensor_call(image_list, attributes, attr_cls, tokens_pipeline)

    # prepare data
    X = []
    y = []
    for i, (_, clip_logits) in enumerate(zip(image_list, image_labels)):
        X.append([extended_clip_logits[1] for extended_clip_logits in clip_logits[1]])
        y_i = y_train.values[i]
        if y_i == "":
            y.append([])
        elif isinstance(y_i, str):
            y.append([y_i])
        else:
            y.append(y_train.values[i])

    X = np.array(X).reshape(len(X), -1)
    y = np.array(y)

    models = ["logistic_regression"]

    model_name = sampler.choice(models, handle=f"{prefix}-model")
    if model_name == "logistic_regression":
        # train logistic regression model
        model = LogisticRegression(random_state=0).fit(X, y)

    return model


def clip_sensor_call(item, attributes: List[str], attr_cls: str, tokens_pipeline):
    """
    Calls a ClipBasedSensor execution.

    :param item: images list
    :param List[str] attributes: attribute class values
    :param str attr_cls: attribute class name
    :return: labels from attributed tokens
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device)

    for tokens in tokens_pipeline:
        text = clip.tokenize(tokens).to(device)

    results = []
    i = 0
    for i in range(0, len(item), min(BATCH_SIZE, len(item) - i)):
        images = []
        for photo_addrs in item[i : min(i + BATCH_SIZE, len(item))]:
            img = Image.open(photo_addrs)
            img_preprocess = preprocess(img)
            img.close()
            images.append(img_preprocess)

        image_input = torch.tensor(np.stack(images)).to(device)
        with torch.no_grad():
            logits_per_image, _ = clip_model(image_input, text)

            batch_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            attribute_probs = [[] for _ in range(len(batch_probs))]
            for k in range(len(batch_probs)):
                image_probs = batch_probs[k]
                for j in range(len(attributes)):
                    attribute_probs[k].append((attributes[j], image_probs[j]))

            attributed_tokens = []
            for h in range(i, min(i + BATCH_SIZE, len(item))):
                attributed_tokens.append(
                    (
                        "image_" + str(i + h % BATCH_SIZE),
                        attribute_probs[h % BATCH_SIZE],
                    )
                )

            results.append(attributed_tokens)

    flatten_results = []
    for batch in results:
        for result in batch:
            flatten_results.append(result)

    return flatten_results


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


def fn(generated: SampleModel, X_test, y_test, stype, attributes, attr_cls, score_func):
    handler: ImageSensorHandler = generated.model
    y_pred = handler.annotate(X_test, stype, attributes, attr_cls)
    score = score_func(y_test, y_pred)
    return score


def build_fn(X_test, y_test, stype, attributes, attr_cls, score_func):
    return partial(
        fn,
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
