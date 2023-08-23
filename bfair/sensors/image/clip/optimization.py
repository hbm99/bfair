from functools import partial

from autogoal.kb import Matrix
from autogoal.sampling import Sampler
from autogoal.search import ConsoleLogger, PESearch

from bfair.methods.autogoal.ensembling.sampling import LogSampler, SampleModel
from bfair.sensors.handler import ImageSensorHandler
from bfair.sensors.image.clip.base import ClipBasedSensor
from bfair.sensors.optimization import build_fn, build_score_fn, evaluate
from bfair.sensors.text.embedding.filters import (BestScoreFilter,
                                                  IdentityFilter,
                                                  LargeEnoughFilter,
                                                  NonEmptyFilter)

MACRO_F1 = "macro-f1"

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
    inspect=False,
    output_stream=None,
):
    score_key = score_key if isinstance(score_key, (list, tuple)) else [score_key]

    tokens_pipeline = get_tokens_pipeline(attr_cls, attributes)

    loggers = [ConsoleLogger()]

    search = PESearch(
        generator_fn =  partial(
            generate, 
            consider_clip_based_sensor = consider_clip_based_sensor, 
            force_clip_based_sensor = force_clip_based_sensor, 
            tokens_pipeline = tokens_pipeline),
        fitness_fn = build_fn(
            X_train,
            y_train,
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
            best_solution,
            X_train,
            y_train,
            attributes,
            attr_cls,
            Matrix
        )
        print("Results @ Training ...", file=output_stream)
        print(counter, file=output_stream)
        print(scores, file=output_stream)
        print(y_pred, file=output_stream, flush=True)

        y_pred, counter, scores = evaluate(
            best_solution,
            X_test,
            y_test,
            attributes,
            attr_cls,
            Matrix
        )
        print("Results @ Testing ....", file=output_stream)
        print(counter, file=output_stream)
        print(scores, file=output_stream)
        print(y_pred, file=output_stream, flush=True)

    return best_solution, best_fn, search


def generate(sampler: Sampler, consider_clip_based_sensor=True, force_clip_based_sensor=False, *, tokens_pipeline):
    """
    Generates a new SampleModel object with the given Sampler.
    """
    sampler = LogSampler(sampler)
    sensors = []
    if force_clip_based_sensor or (consider_clip_based_sensor and sampler.boolean("include-clip-sensor")):
        sensor = get_clip_based_sensor(sampler, tokens_pipeline)
        sensors.append(sensor)

    handler = ImageSensorHandler(sensors, merge=None)
    return SampleModel(sampler, handler)

def get_clip_based_sensor(sampler: LogSampler, tokens_pipeline):
    prefix = "clip-sensor."

    filtering_pipeline = get_filtering_pipeline(sampler, prefix)

    sensor = ClipBasedSensor.build(
        filtering_pipeline=filtering_pipeline,
        tokens_pipeline=tokens_pipeline
    )
    return sensor

def get_tokens_pipeline(attr, attr_values):
    tokens_pipeline = []

    tokens_pipeline.append([attr + ': ' + value for value in attr_values])

    return tokens_pipeline

def get_filtering_pipeline(sampler: LogSampler, prefix):
    filtering_pipeline = []

    filter = get_filter(sampler, allow_none=True, prefix=f"{prefix}filter")
    filtering_pipeline.append(filter)

    filter = NonEmptyFilter()
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
