from .base import Dataset

import pandas as pd
import datasets as db

MALE_VALUE = "Male"
FEMALE_VALUE = "Female"
GENDER_VALUES = [MALE_VALUE, FEMALE_VALUE]


EAST_ASIAN_VALUE = "East Asian"
INDIAN_VALUE = "Indian"
BLACK_VALUE = "Black"
WHITE_VALUE = "White"
MIDDLE_EASTERN_VALUE = "Middle Eastern"
LATINO_HISPANIC_VALUE = "Latino_Hispanic"
SOUTHEAST_ASIAN_VALUE = "Southeast Asian"
RACE_VALUES = [EAST_ASIAN_VALUE, INDIAN_VALUE, BLACK_VALUE, WHITE_VALUE, 
               MIDDLE_EASTERN_VALUE, LATINO_HISPANIC_VALUE, SOUTHEAST_ASIAN_VALUE]

GENDER_COLUMN = "gender"
RACE_COLUMN = 'race'
IMAGE_COLUMN = 'image'
AGE_COLUMN = 'age'

_GENDER_MAP = {
    0: MALE_VALUE,
    1: FEMALE_VALUE
}

_RACE_MAP = {
    0: EAST_ASIAN_VALUE,
    1: INDIAN_VALUE,
    2: BLACK_VALUE,
    3: WHITE_VALUE,
    4: MIDDLE_EASTERN_VALUE,
    5: LATINO_HISPANIC_VALUE,
    6: SOUTHEAST_ASIAN_VALUE
}


def load_dataset(split_seed=None, **kwargs):
    return FairFaceDataset.load(split_seed=split_seed)


class FairFaceDataset(Dataset):
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
        return FairFaceDataset(data=data.sample(3000, random_state=split_seed), split_seed=split_seed)

