import pandas as pd

import datasets as db
from bfair.datasets.build_tools.fairface import (
    create_mixed_dataset,
    get_balanced_by_gender_race as get_balanced,
    save_images_to_disk,
)
from bfair.datasets.fairface import (
    _GENDER_MAP,
    _RACE_MAP,
    AGE_COLUMN,
    GENDER_COLUMN,
    IMAGE_COLUMN,
    RACE_COLUMN,
)

from .base import Dataset

SIZE = 50000
IMAGE_DIR = "datasets/multifairface"


def load_dataset(split_seed=None, **kwargs):
    return MultiFairFaceDataset.load(
        split_seed=split_seed,
        transform_to_paths=kwargs.get("transform_to_paths", True),
        balance_current_representations=kwargs.get(
            "balance_current_representations", False
        ),
    )


class MultiFairFaceDataset(Dataset):
    @classmethod
    def load(cls, split_seed=0, transform_to_paths=True, balance_current_representations=False):
        source = db.load_dataset("HuggingFaceM4/FairFace", split="train")

        df = pd.DataFrame.from_dict(source)
        gender = df[GENDER_COLUMN].apply(lambda x: _GENDER_MAP[x])
        race = df[RACE_COLUMN].apply(lambda x: _RACE_MAP[x])
        data = pd.concat(
            [
                df[IMAGE_COLUMN],
                df[AGE_COLUMN],
                gender.rename(GENDER_COLUMN),
                race.rename(RACE_COLUMN),
            ],
            axis=1,
        )

        # Shuffle the rows of the dataset
        data = data.sample(frac=1, random_state=split_seed).reset_index(drop=True)

        # Create a new dataset with mixed images
        mixed_data = create_mixed_dataset(data, SIZE, split_seed)

        if balance_current_representations:
            mixed_data = get_balanced(mixed_data, split_seed)

        if transform_to_paths:
            save_images_to_disk(mixed_data, IMAGE_DIR)

        return MultiFairFaceDataset(data=mixed_data, split_seed=split_seed)
