
from typing import List, Set, Union

import clip
import numpy as np
import torch
from autogoal.kb import Matrix, SemanticType
from PIL import Image

from bfair.sensors.base import Sensor

BATCH_SIZE = 100000


class ClipBasedSensor(Sensor):
    
    def __init__(self, restricted_to: str | Set[str] = None) -> None:
        super().__init__(restricted_to)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", self.device)
    
    
    def __call__(self, item: List[str], attributes: List[str], attr_cls: str):
        """
        Calls a ClipBasedSensor execution.
        
        :param List[str] item: list containing image addresses
        :param List[str] attributes: possible attribute class values
        :param str attr_class: attribute class
        :return: vectorized function that returns the predicted attribute class values
        """
        tokens = [attr_cls + ': ' + attr for attr in attributes]
        text = clip.tokenize(tokens).to(self.device)
        
        results = []
        for i in range(0, len(item), BATCH_SIZE):
            images = [self.preprocess(Image.open(photo_address for photo_address in item))]
            image_input = torch.tensor(np.stack(images)).to(self.device)
            with torch.no_grad():
                logits_per_image, _ = self.model(image_input, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                results.append(probs)
        
        choices = np.argmax(np.concatenate(results, axis=0), axis=1)
        
        get_label = lambda x: attributes[x]
        v_get_label = np.vectorize(get_label)
        
        return v_get_label(choices)

    def _get_input_type(self) -> SemanticType:
        return Matrix
    