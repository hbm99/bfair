import os

import pandas as pd
from PIL import Image

from bfair.envs import UTK_FACE_DATASET

from .base import Dataset


def load_dataset(path=UTK_FACE_DATASET, split_seed=None):
    return UTKFaceDataset.load(path, split_seed=split_seed)


class UTKFaceDataset(Dataset):
    @classmethod
    def load(cls, path, split_seed=None):
        
        images_paths = os.listdir(path)
        
        images = []
        gender_column = []
        race_column = []
        for image_path in range(len(images_paths)):
            splitted = image_path.split('_')
            images.append(Image.open(image_path))
            gender = splitted[1]
            gender_column.append(gender)
            race = splitted[2]
            race_column.append(race)
            
        images_data = { 'path': images_paths, 'gender': gender_column, 'race': race_column, 'image': images }
            
        data = pd.DataFrame(data = images_data)
        
        return UTKFaceDataset(data)
