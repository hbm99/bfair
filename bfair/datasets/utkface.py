import os

import pandas as pd
from PIL import Image

from bfair.envs import UTK_FACE_DATASET

from .base import Dataset

GENDER_VALUES = ["Male", "Female"]
RACE_VALUES = ["White", "Black", "Asian", "Indian", "Other"]


def load_dataset(path=UTK_FACE_DATASET, split_seed=None):
    return UTKFaceDataset.load(path, split_seed=split_seed)


class UTKFaceDataset(Dataset):
    @classmethod
    def load(cls, path, split_seed=None):
        
        images_paths = os.listdir(path)
        selected_images_paths = []
        images = []
        gender_column = []
        race_column = []
        for i in range(0, len(images_paths), 5):
            if images_paths[i] == '.DS_Store':
                continue
            selected_images_paths.append(images_paths[i])
            splitted = images_paths[i].split('_')
            images.append(Image.open(path + images_paths[i]))
            gender = splitted[1]
            gender_column.append(GENDER_VALUES[int(gender)])
            race = splitted[2]
            race_column.append(RACE_VALUES[int(race)])
            
        images_data = { 'path': selected_images_paths, 'gender': gender_column, 'race': race_column, 'image': images }
            
        data = pd.DataFrame(data = images_data)
        
        return UTKFaceDataset(data, split_seed=split_seed)
