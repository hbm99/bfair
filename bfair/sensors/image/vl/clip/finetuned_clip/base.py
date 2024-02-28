from typing import List, Sequence, Set, Union

import torch
from bfair.sensors.image.vl.clip.base import ClipBasedSensor
from bfair.sensors.text.embedding.filters import Filter

MODEL_PATH = "bfair/sensors/image/vl/clip/finetuned_clip/finetuned_clip.pt"


class FinetunedClipSensor(ClipBasedSensor):
    def __init__(
        self,
        model,
        filtering_pipeline: Sequence[Filter],
        learner,
        logits_to_probs: str,
        tokens_pipeline: Sequence[List[str]],
        restricted_to: Union[str, Set[str]] = None,  # type: ignore
    ) -> None:
        super().__init__(
            model,
            filtering_pipeline=filtering_pipeline,
            learner=learner,
            logits_to_probs=logits_to_probs,
            tokens_pipeline=tokens_pipeline,
            restricted_to=restricted_to,
        )
        self.model.load_state_dict(torch.load(MODEL_PATH))
