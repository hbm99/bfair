import random
import pandas as pd
import numpy as np

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
    GENDER_VALUES,
    IMAGE_COLUMN,
    RACE_COLUMN,
    RACE_VALUES,
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

            def apply_biased_decision_changes(mixed_data, mapping):
                for attr in mapping.keys():

                    def contains_at_least_one(values, target) -> bool:
                        return any(value in target for value in values)

                    column_name = attr + "_biased_decision"
                    mixed_data[column_name] = (
                        mixed_data[attr]
                        .replace(mapping[attr])
                        .apply(
                            lambda x: (
                                1
                                if not isinstance(x, int)
                                and contains_at_least_one(
                                    [
                                        class_value
                                        for class_value, favored in mapping[attr].items()
                                        if favored == 1
                                    ],
                                    x,
                                )
                                else 0
                            )
                        )
                    )

                    # Define the percentages
                    pct_change_1_to_0 = 0.20  # 20% of 1s to 0s
                    pct_change_0_to_1 = 0.30  # 30% of 0s to 1s

                    # Create masks for the changes
                    mask_1s = (mixed_data[column_name] == 1) & (
                        np.random.rand(len(mixed_data)) <= pct_change_1_to_0
                    )
                    mask_0s = (mixed_data[column_name] == 0) & (
                        np.random.rand(len(mixed_data)) <= pct_change_0_to_1
                    )

                    # Apply the masks and make the changes
                    mixed_data.loc[mask_1s, column_name] = 0
                    mixed_data.loc[mask_0s, column_name] = 1

                return mixed_data

            fav_class_mapping = {
                GENDER_COLUMN: {
                    gender: i % 2 for i, gender in enumerate(GENDER_VALUES)
                },
                RACE_COLUMN: {race: i % 2 for i, race in enumerate(RACE_VALUES)},
            }

            # Call the function to apply biased decision changes
            mixed_data = apply_biased_decision_changes(mixed_data, fav_class_mapping)

        return NoisyMultiFairFaceDataset(data=mixed_data, split_seed=split_seed)
