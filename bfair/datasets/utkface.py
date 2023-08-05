import os
import random

import pandas as pd
from PIL import Image

from bfair.envs import UTK_FACE_DATASET

from .base import Dataset

GENDER_VALUES = ["male", "female"]
RACE_VALUES = ["white", "black", "asian", "indian", "other"]
RACE_VALUES_WITH_NOT_RULE = ["white", "black", "asian", "indian", "not white", "not black", "not asian", "not indian"]

GENDER_COLUMN = 'gender'
RACE_COLUMN = 'race'


def load_dataset(path=UTK_FACE_DATASET, split_seed=0):
    return UTKFaceDataset.load(path, split_seed=split_seed)


class UTKFaceDataset(Dataset):
    @classmethod
    def load(cls, path, split_seed=0):
        
        images_paths = os.listdir(path)
        random.seed(split_seed)
        images_paths = random.sample(images_paths, len(images_paths)//5)
        images = []
        gender_column = []
        race_column = []
        for i in range(len(images_paths)):
            if images_paths[i] == '.DS_Store':
                continue
            splitted = images_paths[i].split('_')
            images.append(Image.open(path + images_paths[i]))
            gender = splitted[1]
            gender_column.append(GENDER_VALUES[int(gender)])
            race = splitted[2]
            race_column.append(RACE_VALUES[int(race)])
            
        images_data = { 'path': images_paths, 'gender': gender_column, 'race': race_column, 'image': images }
        
        data = pd.DataFrame(data = images_data)
        
        return UTKFaceDataset(data, split_seed=split_seed)
