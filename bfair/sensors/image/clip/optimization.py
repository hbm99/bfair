from functools import partial

from autogoal.kb import Matrix
from autogoal.sampling import Sampler
from autogoal.search import PESearch

from bfair.methods.autogoal.ensembling.sampling import SampleModel
from bfair.sensors.handler import SensorHandler
from bfair.sensors.image.clip.base import ClipBasedSensor
from bfair.sensors.optimization import build_fn, build_score_fn, compute_errors, compute_scores, evaluate

MACRO_F1 = "macro-f1"

def optimize(
    X_train,
    y_train,
    X_test,
    y_test,
    attributes,
    attr_cls,
    score_key=MACRO_F1,
    *,
    pop_size,
    search_iterations,
    evaluation_timeout,
    memory_limit,
    search_timeout,
    errors="warn",
    log_path,
    inspect=False,
    output_stream=None,
):
    score_key = score_key if isinstance(score_key, (list, tuple)) else [score_key]

    search = PESearch(
        generator_fn = partial(generate), 
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
    best_solution, best_fn = search.run(generations=search_iterations)

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


def generate(sampler: Sampler) -> SampleModel:
    """
    Generates a new SampleModel object with the given Sampler.

    Args:
        sampler (Sampler): The Sampler to use for generating samples.

    Returns:
        SampleModel: A new SampleModel object with the given Sampler.
    """
    handler = SensorHandler([ClipBasedSensor()], merge=None)
    return SampleModel(sampler, handler)

