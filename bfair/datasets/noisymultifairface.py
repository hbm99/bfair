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

CIFAR_IMAGE_COLUMN = "img"
IMAGE_COLUMN = "image"

SIZE = 80000
IMAGE_DIR = "datasets/noisymultifairface"


def load_dataset(split_seed=None, **kwargs):
    return NoisyMultiFairFaceDataset.load(
        split_seed=split_seed,
        transform_to_paths=kwargs.get("transform_to_paths", True),
        balance_current_representations=kwargs.get(
            "balance_current_representations", False
        ),
    )


class NoisyMultiFairFaceDataset(Dataset):
    @classmethod
    def load(
        cls,
        split_seed=0,
        transform_to_paths=True,
        balance_current_representations=False,
    ):
        source_ff = db.load_dataset("HuggingFaceM4/FairFace", split="train")

        df_ff = pd.DataFrame.from_dict(source_ff)
        gender = df_ff[GENDER_COLUMN].apply(lambda x: _GENDER_MAP[x])
        race = df_ff[RACE_COLUMN].apply(lambda x: _RACE_MAP[x])
        df_ff = pd.concat(
            [
                df_ff[IMAGE_COLUMN],
                df_ff[AGE_COLUMN],
                gender.rename(GENDER_COLUMN),
                race.rename(RACE_COLUMN),
            ],
            axis=1,
        )

        source_noisy_dataset = db.load_dataset("cifar100", split="train")
        df_noisy = pd.DataFrame.from_dict(source_noisy_dataset)

        # Remove undesired classifications (people related)
        df_noisy = df_noisy[~df_noisy["fine_label"].isin([2, 11, 35, 46, 98])]
        df_noisy = df_noisy[~df_noisy["coarse_label"].isin([14])]

        new_df_noisy = pd.DataFrame(columns=df_ff.columns)

        new_df_noisy[IMAGE_COLUMN] = df_noisy[CIFAR_IMAGE_COLUMN]

        new_df_noisy = pd.concat([df_ff, new_df_noisy])

        # Shuffle the rows
        new_df_noisy = new_df_noisy.sample(frac=1, random_state=split_seed).reset_index(
            drop=True
        )

        new_df_noisy = new_df_noisy.fillna("")

        mixed_data = create_mixed_dataset(new_df_noisy, SIZE, split_seed)

        if balance_current_representations:
            mixed_data = get_balanced(mixed_data, split_seed)

        if transform_to_paths:
            save_images_to_disk(mixed_data, IMAGE_DIR)

        return NoisyMultiFairFaceDataset(data=mixed_data, split_seed=split_seed)
