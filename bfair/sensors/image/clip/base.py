from typing import List, Sequence, Set, Union

import clip
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
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
        logits_to_probs: str = "sigmoid",
    ) -> None:
        super().__init__(restricted_to)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", self.device)
        self.filtering_pipeline = filtering_pipeline
        self.learner = learner
        self.tokens_pipeline = tokens_pipeline
        self.multi_label_binarizer = None
        self.logits_to_probs = logits_to_probs

    @classmethod
    def build(
        cls,
        filtering_pipeline=(),
        learner=(),
        tokens_pipeline=(),
        logits_to_probs="sigmoid",
    ):
        return cls(filtering_pipeline, learner, tokens_pipeline, logits_to_probs)

    def __call__(self, item, attributes: List[str], attr_cls: str):
        """
        Calls a ClipBasedSensor execution.

        :param item: images list
        :param List[str] attributes: attribute class values
        :param str attr_cls: attribute class name
        :return: labels from attributed tokens
        """

        flatten_results = self.basic_call(item, attributes, attr_cls)

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

    def fit(self, X_train, y_train, attributes, attr_cls):
        if self.learner is None:
            return

        image_list = X_train

        # get labels for each image
        image_labels = self.basic_call(image_list, attributes, attr_cls)

        # prepare data
        X = []
        y = []
        for i, (_, clip_logits) in enumerate(zip(image_list, image_labels)):
            X.append(
                [extended_clip_logits[1] for extended_clip_logits in clip_logits[1]]
            )
            y_i = y_train.values[i]
            if y_i == "":
                y.append({})
            elif isinstance(y_i, str):
                y.append({y_i})
            else:
                y.append(set(y_i))

        X = np.array(X).reshape(len(X), -1)

        mlb = MultiLabelBinarizer()
        self.multi_label_binarizer = mlb

        # Define the outer and inner cross-validation strategies
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

        best_val_score = 0

        for train_index, val_index in outer_cv.split(X, y):
            X_train, X_val = X[train_index], X[val_index]

            mlb.fit(y)
            y_transformed = mlb.transform(y)

            y_train_transformed, y_val_transformed = (
                y_transformed[train_index],
                y_transformed[val_index],
            )

            if not isinstance(self.learner, MultiOutputClassifier):
                model = MultiOutputClassifier(self.learner())
            else:
                model = self.learner

            cv_results = cross_validate(
                model,
                X_train,
                y_train_transformed,
                cv=inner_cv,
                scoring="accuracy",
                return_estimator=True,
            )

            for estimator in cv_results["estimator"]:
                val_score = accuracy_score(y_val_transformed, estimator.predict(X_val))
                if val_score > best_val_score:
                    best_val_score = val_score
                    self.learner = estimator

    def basic_call(self, item, attributes: List[str], attr_cls: str):
        """
        Calls a ClipBasedSensor execution to get `flatten_results`.

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

            with torch.no_grad():
                self.model.eval()
                logits_per_image = torch.empty((0, len(attributes)))
                for image in images:
                    image = image.unsqueeze(0)
                    image = image.to(self.device)
                    logits_per_image = torch.cat(
                        (
                            logits_per_image,
                            self.model(image, text)[0],
                        ),
                        dim=0,
                    )

                if self.logits_to_probs == "sigmoid":
                    batch_probs = (
                        torch.sigmoid(logits_per_image).to(self.device).numpy()
                    )
                else:
                    batch_probs = (
                        logits_per_image.softmax(dim=-1).to(self.device).numpy()
                    )
                    if self.logits_to_probs == "normalize_softmax":
                        max = max(batch_probs)
                        batch_probs = batch_probs / max  # check pending

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

    def _get_input_type(self) -> SemanticType:
        return Matrix
