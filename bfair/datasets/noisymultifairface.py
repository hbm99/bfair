import random
import pandas as pd

import datasets as db
from bfair.datasets.build_tools.fairface import (
    create_balanced_dataset,
    create_mixed_dataset,
    save_images_to_disk,
)
from bfair.datasets.fairface import (
    _GENDER_MAP,
    _RACE_MAP,
    AGE_COLUMN,
    GENDER_COLUMN,
    RACE_COLUMN,
    IMAGE_COLUMN,
    FEMALE_VALUE,
    MALE_VALUE,
    EAST_ASIAN_VALUE,
    INDIAN_VALUE,
    BLACK_VALUE,
    WHITE_VALUE,
    MIDDLE_EASTERN_VALUE,
    LATINO_HISPANIC_VALUE,
    SOUTHEAST_ASIAN_VALUE,
)

from .base import Dataset

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
    )


class NoisyMultiFairFaceDataset(Dataset):
    @classmethod
    def load(
        cls,
        split_seed=0,
        transform_to_paths=True,
        balanced=True,
        decision_columns=False,
    ):
        source_ff = db.load_dataset("HuggingFaceM4/FairFace", split="validation")

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
            mixed_data = create_balanced_dataset(new_df_noisy, SIZE, split_seed)
        else:
            mixed_data = create_mixed_dataset(new_df_noisy, SIZE, split_seed)

        if transform_to_paths:
            save_images_to_disk(mixed_data, IMAGE_DIR)

        if decision_columns:
            random.seed(split_seed)
            mixed_data["random_decision"] = [
                random.randint(0, 1) for _ in range(len(mixed_data))
            ]

            def get_biased_decision(attr, mixed_data, cls_probs):
                biased_decision = []
                for i in range(len(mixed_data)):
                    annotations = mixed_data[attr].iloc[i]
                    prob = 0
                    if not isinstance(annotations, list):
                        annotations = [annotations]
                    for annotation in annotations:
                        prob += cls_probs[annotation]
                    prob /= len(annotations)
                    biased_decision.append(1 if prob > random.randint(0, 1) else 0)
                return biased_decision

            cls_probs = {
                GENDER_COLUMN: {MALE_VALUE: 1, FEMALE_VALUE: 0, "": 0},
                RACE_COLUMN: {
                    EAST_ASIAN_VALUE: 0,
                    INDIAN_VALUE: 0,
                    BLACK_VALUE: 0,
                    WHITE_VALUE: 1,
                    MIDDLE_EASTERN_VALUE: 0,
                    LATINO_HISPANIC_VALUE: 0,
                    SOUTHEAST_ASIAN_VALUE: 0,
                    "": 0,
                },
            }
            for attr in [GENDER_COLUMN, RACE_COLUMN]:
                mixed_data[attr + "_biased_decision"] = get_biased_decision(
                    attr, mixed_data, cls_probs[attr]
                )

        return NoisyMultiFairFaceDataset(data=mixed_data, split_seed=split_seed)
