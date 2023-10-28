from typing import List, Sequence, Set, Union

import torch
from bfair.sensors.image.clip.base import ClipBasedSensor
from bfair.sensors.text.embedding.filters import Filter

MODEL_PATH = "bfair/sensors/image/clip/finetuned_clip/finetuned_clip.pt"


class FinetunedClipSensor(ClipBasedSensor):
    def __init__(
        self,
        filtering_pipeline: Sequence[Filter],
        learner,
        tokens_pipeline: Sequence[List[str]],
        restricted_to: Union[str, Set[str]] = None,
        logits_to_probs: str = "sigmoid",
    ) -> None:
        super().__init__(
            filtering_pipeline, learner, tokens_pipeline, restricted_to, logits_to_probs
        )
        self.model.load_state_dict(torch.load(MODEL_PATH))
