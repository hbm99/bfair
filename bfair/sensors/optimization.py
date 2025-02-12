from bfair.methods.autogoal.ensembling.sampling import (
    LogSampler,
    SampleModel,
    LockedSampler,
)
from bfair.sensors.handler import (
    SensorHandler,
    UnionMerge,
    IntersectionMerge,
    AggregationMerge,
    UniformWeighter,
    ParametricWeighter,
)
# from bfair.sensors.text import (
#     EmbeddingBasedSensor,
#     TextTokenizer,
#     TextSplitter,
#     SentenceTokenizer,
#     NonEmptyFilter,
#     LargeEnoughFilter,
#     BestScoreFilter,
#     NoStopWordsFilter,
#     NonNeutralWordsFilter,
#     IdentityFilter,
#     CountAggregator,
#     ActivationAggregator,
#     UnionAggregator,
#     VotingAggregator,
#     CoreferenceNERSensor,
#     DBPediaSensor,
#     NameGenderSensor,
# )
from autogoal.kb import Text
from autogoal.sampling import Sampler
from autogoal.search import PESearch, ConsoleLogger
from nltk.corpus import stopwords
from statistics import mean
from functools import partial

PRECISION = "precision"
RECALL = "recall"
F1 = "f1"
MACRO_PRECISION = "macro-precision"
MACRO_RECALL = "macro-recall"
MACRO_F1 = "macro-f1"

MICRO_ACC = "micro-accuracy"
MACRO_ACC = "macro-accuracy"


# def optimize(
#     X_train,
#     y_train,
#     X_test,
#     y_test,
#     attributes,
#     attr_cls,
#     score_key=MACRO_F1,
#     consider_embedding_sensor=True,
#     consider_coreference_sensor=True,
#     consider_dbpedia_sensor=True,
#     consider_name_gender_sensor=True,
#     force_embedding_sensors=False,
#     force_coreference_sensor=False,
#     force_dbpedia_sensor=False,
#     force_name_gender_sensor=False,
#     force_union_merge=False,
#     *,
#     pop_size,
#     search_iterations,
#     evaluation_timeout,
#     memory_limit,
#     search_timeout,
#     errors="warn",
#     telegram_token=None,
#     telegram_channel=None,
#     telegram_title="",
#     log_path=None,
#     inspect=False,
#     output_stream=None,
#     language="english",
# ):
#     score_key = score_key if isinstance(score_key, (list, tuple)) else [score_key]

#     loggers = get_loggers(
#         telegram_token=telegram_token,
#         telegram_channel=telegram_channel,
#         telegram_title=telegram_title,
#         log_path=log_path,
#     )

#     search = PESearch(
#         generator_fn=partial(
#             generate,
#             language=language,
#             consider_embedding_sensor=consider_embedding_sensor,
#             consider_coreference_sensor=consider_coreference_sensor,
#             consider_dbpedia_sensor=consider_dbpedia_sensor,
#             consider_name_gender_sensor=consider_name_gender_sensor,
#             force_embedding_sensors=force_embedding_sensors,
#             force_coreference_sensor=force_coreference_sensor,
#             force_dbpedia_sensor=force_dbpedia_sensor,
#             force_name_gender_sensor=force_name_gender_sensor,
#             force_union_merge=force_union_merge,
#         ),
#         fitness_fn=build_fn(
#             X_train,
#             y_train,
#             Text,
#             attributes,
#             attr_cls,
#             score_func=build_score_fn(attributes, score_key),
#         ),
#         maximize=[True] * len(score_key),
#         pop_size=pop_size,
#         evaluation_timeout=evaluation_timeout,
#         memory_limit=memory_limit,
#         search_timeout=search_timeout,
#         errors=errors,
#     )
#     best_solution, best_fn = search.run(generations=search_iterations, logger=loggers)

#     if inspect:
#         y_pred, counter, scores = evaluate(
#             best_solution,
#             X_train,
#             y_train,
#             attributes,
#             attr_cls,
#         )
#         print("Results @ Training ...", file=output_stream)
#         print(counter, file=output_stream)
#         print(scores, file=output_stream)
#         print(y_pred, file=output_stream, flush=True)

#         y_pred, counter, scores = evaluate(
#             best_solution,
#             X_test,
#             y_test,
#             attributes,
#             attr_cls,
#         )
#         print("Results @ Testing ....", file=output_stream)
#         print(counter, file=output_stream)
#         print(scores, file=output_stream)
#         print(y_pred, file=output_stream, flush=True)

#     return best_solution, best_fn, search


def get_loggers(
    *,
    telegram_token=None,
    telegram_channel=None,
    telegram_title="",
    log_path=None,
):
    loggers = [ConsoleLogger()]

    if telegram_token:
        from autogoal.contrib.telegram import TelegramLogger

        telegram = TelegramLogger(
            token=telegram_token,
            name=telegram_title.upper(),
            channel=telegram_channel,
        )
        loggers.append(telegram)

    if log_path:
        from bfair.utils.autogoal import FileLogger

        file_logger = FileLogger(output_path=log_path)
        loggers.append(file_logger)

    return loggers


def evaluate(solution, X, y, attributes, attr_cls, autogoal_type=Text):
    handler: SensorHandler = solution.model
    y_pred = [handler.annotate(item, autogoal_type, attributes, attr_cls) for item in X]
    errors = compute_errors(y, y_pred, attributes)
    scores = compute_scores(errors)
    return y_pred, errors, scores


# def generate(
#     sampler: Sampler,
#     language="english",
#     consider_embedding_sensor=True,
#     consider_coreference_sensor=True,
#     consider_dbpedia_sensor=True,
#     consider_name_gender_sensor=True,
#     force_embedding_sensors=False,
#     force_coreference_sensor=False,
#     force_dbpedia_sensor=False,
#     force_name_gender_sensor=False,
#     force_union_merge=False,
# ):
#     sampler = LogSampler(sampler)

#     sensors = []

#     if force_embedding_sensors or (
#         consider_embedding_sensor and sampler.boolean("include-embedding-sensor")
#     ):
#         sensor = get_embedding_based_sensor(sampler, language)
#         sensors.append(sensor)

#     if force_coreference_sensor or (
#         consider_coreference_sensor and sampler.boolean("include-coreference-sensor")
#     ):
#         sensor = get_coreference_ner_sensor(sampler, language)
#         sensors.append(sensor)

#     if force_dbpedia_sensor or (
#         consider_dbpedia_sensor and sampler.boolean("include-dbpedia-sensor")
#     ):
#         sensor = get_dbpedia_sensor(sampler, language)
#         sensors.append(sensor)

#     if force_name_gender_sensor or (
#         consider_name_gender_sensor and sampler.boolean("include-name-gender-sensor")
#     ):
#         sensor = get_name_gender_sensor(sampler, language)
#         sensors.append(sensor)

#     if len(sensors) > 1:
#         if force_union_merge:
#             merge = UnionMerge()
#         else:
#             merge = get_merger(sampler, len(sensors))
#     else:
#         merge = None

#     handler = SensorHandler(sensors, merge)
#     return SampleModel(sampler, handler)


# def get_embedding_based_sensor(sampler: LogSampler, language):
#     prefix = "embedding-sensor."

#     tokenization_pipeline, plain_mode = get_tokenization_pipeline(sampler, prefix)
#     filtering_pipeline = get_filtering_pipeline(sampler, language, prefix)
#     aggregation_pipeline = get_aggregation_pipeline(sampler, plain_mode, prefix)

#     source = sampler.choice(
#         ["word2vec", "word2vec-debiased"], handle=f"{prefix}embedding-source"
#     )
#     sensor = EmbeddingBasedSensor.build(
#         language=language,
#         source=source,
#         tokenization_pipeline=tokenization_pipeline,
#         filtering_pipeline=filtering_pipeline,
#         aggregation_pipeline=aggregation_pipeline,
#     )
#     return sensor


# def get_tokenization_pipeline(sampler: LogSampler, prefix=""):
#     return (
#         ([TextTokenizer()], True)
#         if sampler.boolean(f"{prefix}plain_mode")
#         else (
#             [
#                 TextSplitter(),
#                 SentenceTokenizer(),
#             ],
#             False,
#         )
#     )


# def get_filtering_pipeline(sampler: LogSampler, language, prefix=""):
#     filtering_pipeline = []

#     if sampler.boolean(f"{prefix}remove-stopwords"):
#         words = stopwords.words(language)
#         filter = NoStopWordsFilter(words)
#         filtering_pipeline.append(filter)

#     filter = get_filter(sampler, allow_none=True, prefix=f"{prefix}filter")
#     filtering_pipeline.append(filter)

#     filter = NonEmptyFilter()
#     filtering_pipeline.append(filter)

#     return filtering_pipeline


# def get_filter(sampler: LogSampler, allow_none: bool, prefix: str):
#     options = ["LargeEnoughFilter", "BestScoreFilter", "NonNeutralWordsFilter"]
#     if allow_none:
#         options.append("None")

#     filter_name = sampler.choice(options, handle=f"{prefix}-filter")

#     if filter_name == "LargeEnoughFilter":
#         norm_threshold = sampler.continuous(
#             0, 1, handle=f"{prefix}-large-norm-threshold"
#         )
#         return LargeEnoughFilter(norm_threshold)

#     elif filter_name == "BestScoreFilter":
#         relative_threshold = sampler.continuous(
#             0, 1, handle=f"{prefix}-best-relative-threshold"
#         )
#         norm_threshold = sampler.continuous(
#             0, 1, handle=f"{prefix}-best-norm-threshold"
#         )
#         return BestScoreFilter(
#             threshold=relative_threshold,
#             zero_threshold=norm_threshold,
#         )

#     elif filter_name == "NonNeutralWordsFilter":
#         relative_threshold = sampler.continuous(
#             0, 1, handle=f"{prefix}-neutral-relative-threshold"
#         )
#         norm_threshold = sampler.continuous(
#             0, 1, handle=f"{prefix}-neutral-norm-threshold"
#         )
#         return NonNeutralWordsFilter(
#             threshold=relative_threshold,
#             zero_threshold=norm_threshold,
#         )

#     elif filter_name == "None" and allow_none:
#         return IdentityFilter()

#     else:
#         raise ValueError(filter_name)


# def get_aggregation_pipeline(sampler: LogSampler, plain_mode, prefix=""):
#     n_iters = 1 if plain_mode else 2
#     aggregation_pipeline = []

#     for i in range(n_iters):
#         aggregator_name = sampler.choice(
#             [
#                 "CountAggregator",
#                 "ActivationAggregator",
#                 "UnionAggregator",
#                 "VotingAggregator",
#             ],
#             handle=f"{prefix}aggretator-{i}",
#         )
#         if aggregator_name == "CountAggregator":
#             filter = get_filter(sampler, allow_none=True, prefix=f"{prefix}count-{i}")
#             aggregator = CountAggregator(attr_filter=filter)

#         elif aggregator_name == "ActivationAggregator":
#             filter = get_filter(
#                 sampler,
#                 allow_none=True,
#                 prefix=f"{prefix}activation-{i}",
#             )

#             activation_name = sampler.choice(
#                 ["max", "sum", "mult"],
#                 handle=f"{prefix}activation-function-{i}",
#             )
#             if activation_name == "max":
#                 activation_func = max
#             elif activation_name == "sum":
#                 activation_func = lambda x, y: x + y
#             elif activation_name == "mult":
#                 activation_func = lambda x, y: x * y
#             else:
#                 raise ValueError(activation_name)

#             aggregator = ActivationAggregator(
#                 activation_func=activation_func, attr_filter=filter
#             )

#         elif aggregator_name == "UnionAggregator":
#             aggregator = UnionAggregator()

#         elif aggregator_name == "VotingAggregator":
#             filter = get_filter(
#                 sampler,
#                 allow_none=True,
#                 prefix=f"{prefix}voting-{i}",
#             )
#             aggregator = VotingAggregator(attr_filter=filter)

#         else:
#             raise ValueError(aggregator_name)

#         aggregation_pipeline.append(aggregator)

#     return aggregation_pipeline


# def get_coreference_ner_sensor(sampler: LogSampler, language):
#     prefix = "coreference-sensor."

#     just_people = sampler.boolean(f"{prefix}just-people")
#     aggregator = get_aggregation_pipeline(
#         sampler,
#         plain_mode=True,
#         prefix=prefix,
#     )[0]
#     sensor = CoreferenceNERSensor.build(
#         language=language,
#         just_people=just_people,
#         aggregator=aggregator,
#     )
#     return sensor


# def get_dbpedia_sensor(sampler: LogSampler, language):
#     prefix = "dbpedia-sensor."

#     just_people = sampler.boolean(f"{prefix}just-people")
#     cutoff = sampler.continuous(min=0, max=1, handle=f"{prefix}cutoff")
#     aggregator = get_aggregation_pipeline(
#         sampler,
#         plain_mode=True,
#         prefix=prefix,
#     )[0]

#     sensor = DBPediaSensor.build(
#         language=language,
#         just_people=just_people,
#         fuzzy_cutoff=cutoff,
#         aggregator=aggregator,
#     )
#     return sensor


# def get_name_gender_sensor(sampler: LogSampler, language):
#     prefix = "name-gender-sensor."

#     just_people = sampler.boolean(f"{prefix}just-people")
#     attention_step = sampler.continuous(min=0, max=1, handle=f"{prefix}attention-step")
#     aggregator = get_aggregation_pipeline(
#         sampler,
#         plain_mode=True,
#         prefix=prefix,
#     )[0]

#     sensor = NameGenderSensor.build(
#         attention_step=attention_step,
#         language=language,
#         just_people=just_people,
#         aggregator=aggregator,
#     )
#     return sensor


# def get_merger(sampler: LogSampler, number_of_sensors):
#     prefix = "merger."

#     merge_mode = sampler.choice(
#         ["union", "intersection", "aggregation"], handle=f"{prefix}mode"
#     )
#     if merge_mode == "union":
#         return UnionMerge()
#     elif merge_mode == "intersection":
#         return IntersectionMerge()
#     elif merge_mode == "aggregation":
#         aggregator = get_aggregation_pipeline(sampler, True, prefix=prefix)[0]

#         weighter_name = sampler.choice(
#             ["uniform", "parametric"], handle=f"{prefix}weighter"
#         )
#         if weighter_name == "uniform":
#             weighter = UniformWeighter()
#         elif weighter_name == "parametric":
#             weights = sampler.multisample(
#                 range(number_of_sensors),
#                 sampler.continuous,
#                 handle=f"{prefix}parametric-weights",
#                 min=0,
#                 max=1,
#             )
#             weights = [weights[i] for i in range(number_of_sensors)]
#             weighter = ParametricWeighter(weights, normalize=False)
#         else:
#             raise ValueError(weighter_name)

#         return AggregationMerge(aggregator, weighter)
#     else:
#         raise ValueError(merge_mode)


def fn(generated: SampleModel, X_test, y_test, stype, attributes, attr_cls, score_func):
    handler: SensorHandler = generated.model
    y_pred = [
        handler.annotate(item, stype, attributes, attr_cls) for item in X_test
    ]
    score = score_func(y_test, y_pred)
    return score

def build_fn(X_test, y_test, stype, attributes, attr_cls, score_func):
    return partial(fn, X_test=X_test, y_test=y_test, stype=stype, attributes=attributes, attr_cls=attr_cls, score_func=score_func)

def score_fn(X, y, attributes, score_keys):
    errors = compute_errors(X, y, attributes)
    scores = compute_scores(errors)
    return tuple(scores[key] for key in score_keys)

def build_score_fn(attributes, score_keys):
    return partial(score_fn, attributes=attributes, score_keys=score_keys)


def compute_errors(y_test, y_pred, attributes):
    ir_counter = {}
    for value in attributes:
        correct_hit = 0
        spurious = 0
        missing = 0
        correct_rejection = 0

        # if true_ann or pred_ann contains values not in attributes, errors are not detected.
        for true_ann, pred_ann in zip(y_test, y_pred):
            if value in true_ann and value not in pred_ann:
                missing += 1
            elif value in pred_ann and value not in true_ann:
                spurious += 1
            elif value in true_ann:
                correct_hit += 1
            else:
                correct_rejection += 1

        ir_counter[value] = (correct_hit, spurious, missing, correct_rejection)

    ac_counter = {}
    for true_ann, pred_ann in zip(y_test, y_pred):
        true_ann = frozenset(true_ann)
        pred_ann = frozenset(pred_ann)

        correct, total = ac_counter.get(true_ann, (0, 0))
        equal = int(true_ann == pred_ann)
        ac_counter[true_ann] = (correct + equal, total + 1)

    return ir_counter, ac_counter


def compute_scores(errors):
    ir_counter, ac_counter = errors

    scores = {}
    per_group = []
    for value, (correct_hit, spurious, missing, _) in ir_counter.items():
        precision = safe_division(correct_hit, correct_hit + spurious)
        recall = safe_division(correct_hit, correct_hit + missing)
        f1 = safe_division(2 * precision * recall, precision + recall)
        scores[value] = group = {
            PRECISION: precision,
            RECALL: recall,
            F1: f1,
        }
        per_group.append(group)

    scores[MACRO_PRECISION] = mean(group[PRECISION] for group in per_group)
    scores[MACRO_RECALL] = mean(group[RECALL] for group in per_group)
    scores[MACRO_F1] = mean(group[F1] for group in per_group)

    total_correct = 0
    total_total = 0
    total_accuracy = 0
    for value, (correct, total) in ac_counter.items():
        total_correct += correct
        total_total += total
        total_accuracy += safe_division(correct, total)

    scores[MICRO_ACC] = safe_division(total_correct, total_total)
    scores[MACRO_ACC] = safe_division(total_accuracy, len(ac_counter))

    return scores


def safe_division(numerator, denominator, default=0):
    return numerator / denominator if denominator else default


# def load(configuration, language="english", root=generate):
#     sampler = LockedSampler(configuration, ensure_handle=True)
#     model = root(sampler, language)
#     return model
