import os
import random
from itertools import combinations

import numpy as np
import pandas as pd
from PIL import Image

import datasets as db
from .base import Dataset


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
RACE_VALUES = [
    EAST_ASIAN_VALUE,
    INDIAN_VALUE,
    BLACK_VALUE,
    WHITE_VALUE,
    MIDDLE_EASTERN_VALUE,
    LATINO_HISPANIC_VALUE,
    SOUTHEAST_ASIAN_VALUE,
]

GENDER_COLUMN = "gender"
RACE_COLUMN = "race"
IMAGE_COLUMN = "image"
AGE_COLUMN = "age"

_GENDER_MAP = {0: MALE_VALUE, 1: FEMALE_VALUE}

_RACE_MAP = {
    0: EAST_ASIAN_VALUE,
    1: INDIAN_VALUE,
    2: BLACK_VALUE,
    3: WHITE_VALUE,
    4: MIDDLE_EASTERN_VALUE,
    5: LATINO_HISPANIC_VALUE,
    6: SOUTHEAST_ASIAN_VALUE,
}

SIZE = 80000
IMAGE_DIR = "datasets/fairface"


def load_dataset(split_seed=None, **kwargs):
    return FairFaceDataset.load(
        split_seed=split_seed, transform_to_paths=kwargs.get("transform_to_paths", True)  # type: ignore
    )


class FairFaceDataset(Dataset):
    @classmethod
    def load(cls, split_seed=0, transform_to_paths=True):
        source = db.load_dataset("HuggingFaceM4/FairFace", split="train")

        df = pd.DataFrame.from_dict(source)  # type: ignore
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

        if transform_to_paths:
            cls.save_images_to_disk(data, IMAGE_DIR)

        return FairFaceDataset(
            data=data.sample(SIZE, random_state=split_seed), split_seed=split_seed
        )

    @staticmethod
    def _get_concat_h_multi_resize(im_list, resample=Image.BICUBIC):
        min_height = min(im.height for im in im_list)
        im_list_resize = [
            im.resize(
                (int(im.width * min_height / im.height), min_height), resample=resample
            )
            for im in im_list
        ]
        total_width = sum(im.width for im in im_list_resize)
        dst = Image.new("RGB", (total_width, min_height))
        pos_x = 0
        for im in im_list_resize:
            dst.paste(im, (pos_x, 0))
            pos_x += im.width
        return dst

    @staticmethod
    def _get_concat_v_multi_resize(im_list, resample=Image.BICUBIC):
        min_width = min(im.width for im in im_list)
        im_list_resize = [
            im.resize(
                (min_width, int(im.height * min_width / im.width)), resample=resample
            )
            for im in im_list
        ]
        total_height = sum(im.height for im in im_list_resize)
        dst = Image.new("RGB", (min_width, total_height))
        pos_y = 0
        for im in im_list_resize:
            dst.paste(im, (0, pos_y))
            pos_y += im.height
        return dst

    @staticmethod
    def _get_concat_tile_resize(im_list_2d, resample=Image.BICUBIC):
        im_list_v = [
            FairFaceDataset._get_concat_h_multi_resize(im_list_h, resample=resample)
            for im_list_h in im_list_2d
        ]
        return FairFaceDataset._get_concat_v_multi_resize(im_list_v, resample=resample)

    @staticmethod
    def _concat_images(image_list):
        return FairFaceDataset._get_concat_tile_resize(image_list)

    @staticmethod
    def _merge_attribute_values(attribute, row_i, row_j):
        attribute_value_i = row_i[attribute]
        attribute_value_j = row_j[attribute]

        if attribute_value_i == "" and attribute_value_j == "":
            return ""
        if attribute_value_i == "":
            return attribute_value_j
        if attribute_value_j == "":
            return attribute_value_i

        if (
            isinstance(attribute_value_i, list)
            and isinstance(attribute_value_j, str)
            and attribute_value_j not in attribute_value_i
        ):
            attribute_value_i.append(attribute_value_j)

        elif (
            isinstance(attribute_value_i, str)
            and isinstance(attribute_value_j, list)
            and attribute_value_i not in attribute_value_j
        ):
            attribute_value_j.append(attribute_value_i)
            attribute_value_i = attribute_value_j

        elif isinstance(attribute_value_i, list) and isinstance(
            attribute_value_j, list
        ):
            attribute_value_i = list(set(attribute_value_i + attribute_value_j))

        elif (
            isinstance(attribute_value_i, str)
            and isinstance(attribute_value_j, str)
            and attribute_value_i != attribute_value_j
        ):
            attribute_value_i = [attribute_value_i, attribute_value_j]

        if isinstance(attribute_value_i, list) and len(attribute_value_i) == 1:
            attribute_value_i = attribute_value_i[0]

        return attribute_value_i

    @staticmethod
    def _contains(annotations, representation):
        for attr_clss in representation:
            if attr_clss not in annotations:
                return False
        return True

    @classmethod
    def create_mixed_dataset(cls, data, size, split_seed):
        random.seed(split_seed)
        mixed_data = pd.DataFrame(columns=data.columns)
        num_rows = len(data)
        for i in range(size):
            row_i = data.iloc[i]

            image_list = [row_i[IMAGE_COLUMN]]
            rows_to_concat = random.randint(0, 2)
            for _ in range(rows_to_concat):
                row_j = data.iloc[random.randint(0, num_rows - 1)]

                image_list.append(row_j[IMAGE_COLUMN])

                for attribute in [GENDER_COLUMN, RACE_COLUMN, AGE_COLUMN]:
                    row_i[attribute] = cls._merge_attribute_values(
                        attribute, row_i, row_j
                    )

            # Shuffle the list
            random.shuffle(image_list)

            # Determine the number of chunks
            num_chunks = random.randint(1, len(image_list))

            # Calculate the size of each chunk
            chunk_size = len(image_list) // num_chunks

            row_i[IMAGE_COLUMN] = cls._concat_images(
                [
                    image_list[i : i + chunk_size]
                    for i in range(0, len(image_list), chunk_size)
                ]
            )
            image = row_i[IMAGE_COLUMN]
            sqrWidth = np.ceil(np.sqrt(image.size[0] * image.size[1])).astype(int)
            im_resize = image.resize((sqrWidth, sqrWidth))
            row_i[IMAGE_COLUMN] = im_resize
            mixed_data = mixed_data.append(row_i, ignore_index=True)

        return mixed_data

    @classmethod
    def create_balanced_dataset(cls, data, size, split_seed):
        random.seed(split_seed)

        size_per_representation = size // (
            (2 ** len(RACE_VALUES) - 1) * (2 ** len(GENDER_VALUES) - 1)
        )
        gender_representations = [
            list(combinations(GENDER_VALUES, i))
            for i in range(1, len(GENDER_VALUES) + 1)
        ]
        gender_representations = [
            item for sublist in gender_representations for item in sublist
        ]
        race_representations = [
            list(combinations(RACE_VALUES, i)) for i in range(1, len(RACE_VALUES) + 1)
        ]
        race_representations = [
            item for sublist in race_representations for item in sublist
        ]

        balanced_data = pd.DataFrame(columns=data.columns)

        empty_gender_race_repr_df = data[
            (data[GENDER_COLUMN].apply(lambda x: x == ""))
            & (data[RACE_COLUMN].apply(lambda x: x == ""))
        ]
        for gender_representation in gender_representations:
            for race_representation in race_representations:
                gender_repr_df = data[
                    data[GENDER_COLUMN].apply(lambda x: x in gender_representation)
                ]
                repr_df = gender_repr_df[
                    gender_repr_df[RACE_COLUMN].apply(
                        lambda x: x in race_representation
                    )
                ]

                for i in range(size_per_representation):
                    row_i = repr_df.iloc[i]
                    image_list = [row_i[IMAGE_COLUMN]]

                    rows_to_concat = random.randint(0, 2)
                    index = 0
                    all_represented = False
                    while index < rows_to_concat or not all_represented:
                        index += 1
                        row_j = repr_df.iloc[random.randint(0, len(repr_df) - 1)]
                        image_list.append(row_j[IMAGE_COLUMN])

                        for attr in [GENDER_COLUMN, RACE_COLUMN, AGE_COLUMN]:
                            row_i[attr] = cls._merge_attribute_values(
                                attr, row_i, row_j
                            )
                        if cls._contains(
                            row_i[GENDER_COLUMN], gender_representation
                        ) and cls._contains(row_i[RACE_COLUMN], race_representation):
                            all_represented = True

                    # Shuffle the list
                    random.shuffle(image_list)

                    # Determine the number of chunks
                    num_chunks = random.randint(1, len(image_list))

                    # Calculate the size of each chunk
                    chunk_size = len(image_list) // num_chunks

                    row_i[IMAGE_COLUMN] = cls._concat_images(
                        [
                            image_list[i : i + chunk_size]
                            for i in range(0, len(image_list), chunk_size)
                        ]
                    )

                    # Resize image for compatibility with CLIP
                    image = row_i[IMAGE_COLUMN]
                    sqrWidth = np.ceil(np.sqrt(image.size[0] * image.size[1])).astype(
                        int
                    )
                    im_resize = image.resize((sqrWidth, sqrWidth))
                    row_i[IMAGE_COLUMN] = im_resize

                    balanced_data = balanced_data.append(row_i, ignore_index=True)

        repr_df = empty_gender_race_repr_df
        for i in range(size_per_representation):
            row_i = repr_df.iloc[i]
            image_list = [row_i[IMAGE_COLUMN]]
            rows_to_concat = random.randint(0, 2)
            for _ in range(rows_to_concat):
                row_j = repr_df.iloc[random.randint(0, len(repr_df) - 1)]

                image_list.append(row_j[IMAGE_COLUMN])

            # Shuffle the list
            random.shuffle(image_list)

            # Determine the number of chunks
            num_chunks = random.randint(1, len(image_list))

            # Calculate the size of each chunk
            chunk_size = len(image_list) // num_chunks

            row_i[IMAGE_COLUMN] = cls._concat_images(
                [
                    image_list[i : i + chunk_size]
                    for i in range(0, len(image_list), chunk_size)
                ]
            )

            # Resize image for compatibility with CLIP
            image = row_i[IMAGE_COLUMN]
            sqrWidth = np.ceil(np.sqrt(image.size[0] * image.size[1])).astype(int)
            im_resize = image.resize((sqrWidth, sqrWidth))
            row_i[IMAGE_COLUMN] = im_resize

            balanced_data = balanced_data.append(row_i, ignore_index=True)

        return balanced_data

    @classmethod
    def save_images_to_disk(cls, data, image_dir):
        # Create the image directory if it doesn't exist
        os.makedirs(image_dir, exist_ok=True)

        # Save the images to disk and replace the image column with the image paths
        for i, row in data.iterrows():
            img = row[IMAGE_COLUMN]
            img_path = os.path.join(image_dir, f"{i}.jpg")
            img.save(img_path)
            data.at[i, IMAGE_COLUMN] = img_path
