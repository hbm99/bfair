from typing import List, Sequence, Set, Union

from autogoal.kb import Matrix, SemanticType

from bfair.sensors.base import Sensor
from bfair.sensors.text.embedding.filters import Filter

BATCH_SIZE = 64


class VisionLanguageBasedSensor(Sensor):
    def __init__(
        self,
        model,
        filtering_pipeline: Sequence[Filter],
        learner,
        logits_to_probs: str,
        tokens_pipeline: Sequence[List[str]],
        restricted_to: Union[str, Set[str]] = None,  # type: ignore
    ) -> None:
        super().__init__(restricted_to)
        self.model, self.preprocess = model
        self.filtering_pipeline = filtering_pipeline
        self.learner = learner
        self.tokens_pipeline = tokens_pipeline
        self.logits_to_probs = logits_to_probs

    @classmethod
    def build(
        cls,
        model=(),
        filtering_pipeline=(),
        learner=(),
        logits_to_probs="sigmoid",
        tokens_pipeline=(),
    ):
        return cls(model, filtering_pipeline, learner, logits_to_probs, tokens_pipeline)

    def __call__(self, item, attributes: List[str], attr_cls: str):
        """
        Calls a VisionLanguageSensor execution.

        :param item: images list
        :param List[str] attributes: attribute class values
        :param str attr_cls: attribute class name
        :return: labels from attributed tokens
        """
        pass

    def _get_input_type(self) -> SemanticType:
        return Matrix
