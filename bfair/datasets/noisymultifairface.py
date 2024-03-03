import random
import pandas as pd
import datasets as db

from bfair.datasets.fairface import (
    AGE_COLUMN,
    BLACK_VALUE,
    EAST_ASIAN_VALUE,
    FEMALE_VALUE,
    GENDER_COLUMN,
    IMAGE_COLUMN,
    INDIAN_VALUE,
    LATINO_HISPANIC_VALUE,
    MALE_VALUE,
    MIDDLE_EASTERN_VALUE,
    RACE_COLUMN,
    SOUTHEAST_ASIAN_VALUE,
    WHITE_VALUE,
    _GENDER_MAP,
    _RACE_MAP,
    FairFaceDataset,
)


CIFAR_IMAGE_COLUMN = "img"
IMAGE_COLUMN = "image"

SIZE = 20000
IMAGE_DIR = "datasets/noisymultifairface"


def load_dataset(split_seed=None, **kwargs):
    return NoisyMultiFairFaceDataset.load(
        split_seed=split_seed,
        transform_to_paths=kwargs.get("transform_to_paths", True),
        balanced=kwargs.get("balanced", True),
        decision_columns=kwargs.get("decision_columns", False),
        split=kwargs.get("split", "train"),
    )


class NoisyMultiFairFaceDataset(FairFaceDataset):
    @classmethod
    def load(
        cls,
        split_seed=0,
        transform_to_paths=True,
        balanced=True,
        decision_columns=False,
        split="train",
    ):
        source_ff = db.load_dataset("HuggingFaceM4/FairFace", split=split)

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

        source_noisy_dataset = db.load_dataset("cifar100", split="test")
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

        if balanced:
            mixed_data = cls.create_balanced_dataset(new_df_noisy, SIZE, split_seed)
        else:
            mixed_data = cls.create_mixed_dataset(new_df_noisy, SIZE, split_seed)

        if transform_to_paths:
            cls.save_images_to_disk(mixed_data, IMAGE_DIR)

        if decision_columns:
            cls.add_decisions(split_seed, mixed_data)

        return NoisyMultiFairFaceDataset(data=mixed_data, split_seed=split_seed)
