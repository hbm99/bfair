
from typing import Sequence, Tuple

from bfair.sensors.text.embedding.filters import BestScoreFilter


class NotRuleFilter(BestScoreFilter):
    """
    Filter that returns the best attribute after checking the "not rule". 
    If the positive probs are greater than the negative probs of an attribute value 
    for all possible attribute values, 
    then None is added to token, wich means there are not valid values. 
    (different to empty output wich means not close enough scores)
    """
    def __call__(self, attributed_tokens: Sequence[Tuple[str, Sequence[Tuple[str, float]]]]) -> Sequence[Tuple[str, Sequence[Tuple[str, float]]]]:
        output = []
        for token, attributes in attributed_tokens:

            attribute_scores = {}
            for attr, score in attributes:
                if attr.split()[0] != "not":
                    attribute_scores[attr] = score

            for attr, score in attributes:
                if attr.split()[0] == "not":
                    attr_value = attr.split()[1]
                    if attr_value in attribute_scores and score > attribute_scores[attr_value]:
                        attribute_scores.pop(attr_value)

            # attribute_scores only contains the score if the positive value is greater than the negative value

            if len(attribute_scores) == 0:
                output.append((token, None))
                continue
                    
            max_score = max((score for attr, score in attributes if attr.split()[0] != 'not'), default=float("-inf"))

            best_attributes = [(attr, score) for attr, score in attributes
                               if attr.split()[0] != 'not' and 
                               self._is_close_enough(score, max_score)]
            
            output.append((token, best_attributes))
        return output
