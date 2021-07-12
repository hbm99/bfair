from typing import Callable, List

from bfair.metrics import build_oracle_output_matrix, disagreement

from autogoal.kb import Supervised, VectorCategorical
from autogoal.ml import AutoML
from autogoal.search import PESearch


class AutoGoalDiversifier:
    def __init__(self, *, input, n_classifiers: int, maximize=True, **automl_kwargs):
        self.n_classifiers = n_classifiers
        self.maximize = maximize

        input = (
            input + (Supervised[VectorCategorical],)
            if isinstance(input, tuple)
            else (input, Supervised[VectorCategorical])
        )

        self._automl = AutoML(
            input=input,
            output=VectorCategorical,
            search_algorithm=PESearch,
            number_of_pipelines=n_classifiers,
            maximize=maximize,
            **automl_kwargs,
        )

    def __call__(self, X, y, **search_kwargs):
        return self._build_base_classifiers(X, y, **search_kwargs)

    def _build_base_classifiers(self, X, y, **search_kwargs):
        automl = self._automl
        ranking_fn = self.make_ranking_fn(
            X,
            y,
            top_cut=self.n_classifiers,
            maximize=self.maximize,
        )

        automl.fit(X, y, ranking_fn=ranking_fn, **search_kwargs)
        return automl.top_pipelines_, automl.top_pipelines_scores_

    @staticmethod
    def make_ranking_fn(
        X, y, *, top_cut=None, maximize=True
    ) -> Callable[[List, List[float]], List[float]]:
        def ranking_fn(solutions, fns):
            if top_cut is not None and len(solutions) <= top_cut:
                return fns

            n_solutions = len(solutions)
            ranking = [None] * len(solutions)

            # COMPUTE PREDICTIONS AND FIND TOP PERFORMING PIPELINE'S INDEX
            best = None
            best_fn = None
            predictions = []
            valid_solutions = []

            for i in range(n_solutions):
                pipeline = solutions[i]
                fn = fns[i]

                try:
                    pipeline.send("train")
                    pipeline.run(X, y)
                    pipeline.send("eval")
                    y_pred = pipeline.run(X, None)

                    predictions.append(y_pred)
                    valid_solutions.append(pipeline)

                    if (
                        best_fn is None
                        or (fn > best_fn and maximize)
                        or (fn < best_fn and not maximize)
                    ):
                        best = i
                        best_fn = fn
                except Exception:
                    ranking[i] = -1

            if (
                top_cut is not None
                and len(valid_solutions) <= top_cut
                or len(valid_solutions) < 3
            ):
                return fns

            # COMPUTE DIVERSITY BETWEEN PAIRS OF PIPELINES
            oracle_matrix = build_oracle_output_matrix(y, predictions)
            diversity_matrix = disagreement(oracle_matrix)

            # RANK PIPELINES GREEDY (root: best_fn, step: diversity)
            # - accuracy is been ignored
            n_valid_solutions = len(valid_solutions)
            ranking[best] = n_valid_solutions
            selected = [best]

            def compute_aggregated_diversity(pipeline_index):
                return diversity_matrix[selected, pipeline_index].sum()

            for rank in range(n_valid_solutions - 1, 1, -1):
                best = None
                best_diversity = float("-inf")
                for i in range(n_solutions):
                    if ranking[i] is not None:
                        continue
                    diversity = compute_aggregated_diversity(i)
                    if diversity > best_diversity:
                        best = i
                        best_diversity = diversity
                ranking[best] = rank
                selected.append(best)

            last = next(i for i in range(n_solutions) if ranking[i] is None)
            ranking[last] = 1

            return ranking

        return ranking_fn