from random import random, sample, randint

import pandas as pd

import datasets as db
from bfair.datasets.build_tools.multifairface import (concat_images,
                                                      merge_attribute_values)
from bfair.datasets.fairface import (_GENDER_MAP, _RACE_MAP, AGE_COLUMN,
                                     GENDER_COLUMN, IMAGE_COLUMN, RACE_COLUMN)

from .base import Dataset


def load_dataset(split_seed=None, **kwargs):
    return MultiFairFaceDataset.load(split_seed=split_seed)


class MultiFairFaceDataset(Dataset):

    @classmethod
    def load(cls, split_seed = 0):
        source = db.load_dataset("HuggingFaceM4/FairFace", split="train")

        df = pd.DataFrame.from_dict(source)
        gender = df[GENDER_COLUMN].apply(lambda x: _GENDER_MAP[x])
        race = df[RACE_COLUMN].apply(lambda x: _RACE_MAP[x])
        data = pd.concat(
            [
                df[IMAGE_COLUMN],
                df[AGE_COLUMN],
                gender.rename(GENDER_COLUMN),
                race.rename(RACE_COLUMN)
            ],
            axis=1
        )

        # Shuffle the rows of the dataset
        data = data.sample(frac=1, random_state=split_seed).reset_index(drop=True)

        # Create a new dataset with mixed images
        mixed_data = pd.DataFrame(columns=data.columns)
        num_rows = len(data)
        for i in range(num_rows):
            row_i = data.iloc[i]
            
            image_list = [row_i[IMAGE_COLUMN]]
            rows_to_concat = randint(0, 6)
            for _ in range(rows_to_concat):
                row_j = data.iloc[randint(0, num_rows - 1)]

                image_list.append(row_j[IMAGE_COLUMN])
                
                for attribute in [GENDER_COLUMN, RACE_COLUMN, AGE_COLUMN]:
                    row_i[attribute] = merge_attribute_values(attribute, row_i, row_j)
                
            row_i[IMAGE_COLUMN] = concat_images([sample(image_list, randint(1, len(image_list))) for _ in range(3)])
            mixed_data = mixed_data.append(row_i, ignore_index=True)
        
        return MultiFairFaceDataset(data=mixed_data.sample(30000, random_state=split_seed), split_seed=split_seed)

    

        
        

