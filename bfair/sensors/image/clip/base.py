
from typing import List, Set, Union

import clip
import numpy as np
import torch
from autogoal.kb import Matrix, SemanticType

from bfair.sensors.base import Sensor
from bfair.sensors.text.embedding.filters import Filter

BATCH_SIZE = 64


class ClipBasedSensor(Sensor):
    
    def __init__(self, filter: Filter, restricted_to: Union[str, Set[str]] = None) -> None:
        super().__init__(restricted_to)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", self.device)
        self.filter = filter

    def __call__(self, item, attributes: List[str], attr_cls: str):
        """
        Calls a ClipBasedSensor execution.
        
        :param item: list containing images
        :param List[str] attributes: possible attribute class values
        :param str attr_class: attribute class
        :return: vectorized function that returns the predicted attribute class values
        """
        # tokens = [attr_cls + ': ' + attr for attr in attributes]
        tokens = ['This is a person of ' + attr + ' ' + attr_cls for attr in attributes]
        text = clip.tokenize(tokens).to(self.device)
        
        results = []
        i = 0
        for i in range(0, len(item), min(BATCH_SIZE, len(item) - i)):
            images = [self.preprocess(photo) for photo in item[i: min(i + BATCH_SIZE, len(item))]]
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
                    attributed_tokens.append(('image_' + str(i + h % BATCH_SIZE), 
                                              attribute_probs[h % BATCH_SIZE]))
                
                results.append(attributed_tokens)
        
        flatten_results = []
        for batch in results:
            for result in batch:
                flatten_results.append(result)
        
        attributed_tokens = self.filter(flatten_results)
        
        return attributed_tokens

    def _get_input_type(self) -> SemanticType:
        return Matrix
    