from typing import List, Sequence, Set, Union

import clip
import numpy as np
import torch
from autogoal.kb import Matrix, SemanticType
from PIL import Image

from bfair.sensors.base import Sensor
from bfair.sensors.text.embedding.filters import Filter

BATCH_SIZE = 64


class ClipBasedSensor(Sensor):
    def __init__(
        self,
        filtering_pipeline: Sequence[Filter],
        learner,
        tokens_pipeline: Sequence[List[str]],
        restricted_to: Union[str, Set[str]] = None,
    ) -> None:
        super().__init__(restricted_to)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", self.device)
        self.filtering_pipeline = filtering_pipeline
        self.learner, self.multi_label_binarizer = learner
        self.tokens_pipeline = tokens_pipeline

    @classmethod
    def build(cls, filtering_pipeline=(), learner=(), tokens_pipeline=()):
        return cls(filtering_pipeline, learner, tokens_pipeline)

    def __call__(self, item, attributes: List[str], attr_cls: str):
        """
        Calls a ClipBasedSensor execution.

        :param item: images list
        :param List[str] attributes: attribute class values
        :param str attr_cls: attribute class name
        :return: labels from attributed tokens
        """
        for tokens in self.tokens_pipeline:
            text = clip.tokenize(tokens).to(self.device)

        results = []
        i = 0
        for i in range(0, len(item), min(BATCH_SIZE, len(item) - i)):
            images = []
            for photo_addrs in item[i : min(i + BATCH_SIZE, len(item))]:
                img = Image.open(photo_addrs)
                img_preprocess = self.preprocess(img)
                img.close()
                images.append(img_preprocess)

            image_input = torch.tensor(np.stack(images)).to(self.device)
            with torch.no_grad():
                logits_per_image, _ = self.model(image_input, text)

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

        selected_results = (
            self.get_filtered(flatten_results)
            if self.filtering_pipeline
            else self.get_learned(flatten_results)
        )

        return selected_results

    def get_learned(self, results):
        X = []
        for labels in results:
            X.append([extended_labels[1] for extended_labels in labels[1]])
        y_pred_transformed = self.learner.predict(X)
        y_pred = self.multi_label_binarizer.inverse_transform(y_pred_transformed)
        return list(y_pred)

    def get_filtered(self, results):
        attributed_tokens = []
        for filter in self.filtering_pipeline:
            attributed_tokens = filter(results)

        labels_from_attr_tokens = []
        for _, labels_values_pair in attributed_tokens:
            labels_from_attr_tokens.append([labels for labels, _ in labels_values_pair])

        return labels_from_attr_tokens

    def _get_input_type(self) -> SemanticType:
        return Matrix
