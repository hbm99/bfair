
import json

from bfair.datasets import load_utkface
from bfair.datasets.utkface import (GENDER_VALUES, RACE_VALUES,
                                    RACE_VALUES_WITH_NOT_RULE)
from bfair.metrics import exploded_accuracy_disparity
from bfair.sensors.base import P_GENDER, P_RACE
from bfair.sensors.image.clip.base import ClipBasedSensor
from bfair.sensors.image.clip.filters import NotRuleFilter
from bfair.sensors.optimization import compute_errors, compute_scores
from bfair.sensors.text.embedding.filters import BestScoreFilter

POSITIVE_TARGETS = {P_GENDER: 'Male', P_RACE: 'White'}


def evaluate(values, attribute, phrases=None, filter=BestScoreFilter(), not_rule=False):

    phrases = [attribute + ': ' + value for value in values[1]] if not_rule else [attribute + ': ' + attr for attr in values] if not phrases else phrases
    
    clip_sensor = ClipBasedSensor(filter)
    print("Loaded!")

    dataset = load_utkface(split_seed=0)
    
    X = dataset.data['image']
    y = dataset.data[attribute]

    predictions = clip_sensor(X, values[1] if not_rule else values, phrases)

    new_y, new_predictions = _new_format(y, predictions)

    scores = _get_scores(values[0] if not_rule else values, new_y, new_predictions)
    
    fairness = _get_accuracy_disparity(attribute, dataset, new_predictions)

    return scores, fairness

def _get_fairness_to_json(fairness):
    new_fairness = {}
    for key in fairness[1].keys():
        value = fairness[1][key]
        word_accum = ''
        for word in key:
            word_accum += word + ' '
        new_fairness[word_accum] = value
    
    new_fairness = (fairness[0], new_fairness)
    return new_fairness

def _get_accuracy_disparity(attribute, dataset, new_predictions):
    fairness = exploded_accuracy_disparity(
        data=dataset.data,
        protected_attributes= [P_GENDER, P_RACE],
        target_attribute=attribute,
        target_predictions= [new_predictions[i][0] for i in range(len(new_predictions))],
        positive_target=POSITIVE_TARGETS[attribute],
        return_probs=True,
    )
    print(dataset.data)
    print("Fairness:", fairness)
    return _get_fairness_to_json(fairness)

def _get_scores(values, new_y, new_predictions):
    errors = compute_errors(new_y, new_predictions, values)
    print(errors)

    scores = compute_scores(errors)
    print(scores)
    return scores

def _new_format(y, predictions):
    new_y = [[] for _ in range(len(y))]
    for i in range(len(y)):
        new_y[i].append(y.values[i])

    new_predictions = [[] for _ in range(len(predictions))]
    for i in range(len(predictions)):
        pred_i = predictions[i][1]
        if not pred_i:
            new_predictions[i].append('other')
            continue
        for j in range(len(pred_i)):
            new_predictions[i].append(pred_i[j][0])
    return new_y, new_predictions


def _get_phrases(attr_values, attr):
    phrases_types = [
            (attr + ": __attr__", [attr + ': ' + value for value in attr_values]), 
            ('This is a person of __attr__ ' + attr, ['This is a person of ' + value + ' ' + attr for value in attr_values]),
            ('This is a person of ' + attr + ' __attr__', ['This is a person of ' + attr + ' ' + value for value in attr_values]),
            ('A person of __attr__ ' + attr, ['A person of ' + value + ' ' + attr for value in attr_values]),
            ('A person of ' + attr + ' __attr__', ['A person of ' + attr + ' ' + value for value in attr_values]),
            ('A __attr__ ' + attr + ' person', ['A ' + value + ' ' + attr + ' person' for value in attr_values]),
            ('An image of a person of __attr__ ' + attr, ['An image of a person of ' + value + ' ' + attr for value in attr_values])
                        ]
                    
    return phrases_types

def run_experiment(attribute_tuples):
    json_results = []
    for attr_tuple in attribute_tuples:
        not_rule = attr_tuple[0]
        attr_values = attr_tuple[1]
        attr = attr_tuple[2]

        # phrases
        phrases_types = _get_phrases(attr_values[1], attr) if not_rule else _get_phrases(attr_values, attr)
    
        for phrases in phrases_types:
            phrase_type = phrases[0]
            phrases_list = phrases[1]

            scores, fairness = evaluate(attr_values, attr, phrases_list, NotRuleFilter(), not_rule)

            json_results.append(
                {
                    phrase_type: 
                    {
                        'scores': scores,
                        'fairness': fairness
                    }
                }
            )
            
    return json_results

def _write_json_file(json_results, filename = 'scores__accuracy_disparity__evaluation'):
    '''
    Write scores and fairness to JSON file.
    '''
    with open('results/clip_based_sensor/' + filename + '.json', 'a') as f:
        f.write(json.dumps(json_results, indent=4))

if __name__ == "__main__":

    attribute_tuples = [(True, [RACE_VALUES, RACE_VALUES_WITH_NOT_RULE], P_RACE), (False, GENDER_VALUES, P_GENDER)]

    results = run_experiment(attribute_tuples)
    
    _write_json_file(results, 'scores__accuracy_disparity__evaluation_with_not_rule')