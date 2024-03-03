import pandas as pd
import datasets as db

from bfair.datasets.fairface import (
    _GENDER_MAP,
    _RACE_MAP,
    AGE_COLUMN,
    GENDER_COLUMN,
    IMAGE_COLUMN,
    RACE_COLUMN,
    FairFaceDataset,
)


SIZE = 50000
IMAGE_DIR = "datasets/multifairface"


def load_dataset(split_seed=None, **kwargs):
    return MultiFairFaceDataset.load(
        split_seed=split_seed,
        transform_to_paths=kwargs.get("transform_to_paths", True),
        balanced=kwargs.get("balanced", True),
        split=kwargs.get("split", "train"),
    )


class MultiFairFaceDataset(FairFaceDataset):
    @classmethod
    def load(cls, split_seed=0, transform_to_paths=True, balanced=True, split="train"):
        source = db.load_dataset("HuggingFaceM4/FairFace", split=split)

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

        if balanced:
            mixed_data = cls.create_balanced_dataset(data, SIZE, split_seed)
        else:
            mixed_data = cls.create_mixed_dataset(data, SIZE, split_seed)

        if transform_to_paths:
            cls.save_images_to_disk(mixed_data, IMAGE_DIR)

        return MultiFairFaceDataset(data=mixed_data, split_seed=split_seed)
