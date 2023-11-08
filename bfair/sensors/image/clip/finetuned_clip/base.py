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
        logits_to_probs: str,
        tokens_pipeline: Sequence[List[str]],
        restricted_to: Union[str, Set[str]] = None,
    ) -> None:
        super().__init__(
            filtering_pipeline, learner, logits_to_probs, tokens_pipeline, restricted_to
        )
        self.model.load_state_dict(torch.load(MODEL_PATH))
