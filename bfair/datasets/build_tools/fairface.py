import os
import random

import pandas as pd
from PIL import Image

GENDER_COLUMN = "gender"
RACE_COLUMN = "race"
IMAGE_COLUMN = "image"
AGE_COLUMN = "age"


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


def _get_concat_v_multi_resize(im_list, resample=Image.BICUBIC):
    min_width = min(im.width for im in im_list)
    im_list_resize = [
        im.resize((min_width, int(im.height * min_width / im.width)), resample=resample)
        for im in im_list
    ]
    total_height = sum(im.height for im in im_list_resize)
    dst = Image.new("RGB", (min_width, total_height))
    pos_y = 0
    for im in im_list_resize:
        dst.paste(im, (0, pos_y))
        pos_y += im.height
    return dst


def _get_concat_tile_resize(im_list_2d, resample=Image.BICUBIC):
    im_list_v = [
        _get_concat_h_multi_resize(im_list_h, resample=resample)
        for im_list_h in im_list_2d
    ]
    return _get_concat_v_multi_resize(im_list_v, resample=resample)


def concat_images(image_list):
    return _get_concat_tile_resize(image_list)


def merge_attribute_values(attribute, row_i, row_j):
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

    elif isinstance(attribute_value_i, list) and isinstance(attribute_value_j, list):
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


def create_mixed_dataset(data, size, split_seed):
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
                row_i[attribute] = merge_attribute_values(attribute, row_i, row_j)

        # Shuffle the list
        random.shuffle(image_list)

        # Determine the number of chunks
        num_chunks = random.randint(1, len(image_list))

        # Calculate the size of each chunk
        chunk_size = len(image_list) // num_chunks

        row_i[IMAGE_COLUMN] = concat_images(
            [
                image_list[i : i + chunk_size]
                for i in range(0, len(image_list), chunk_size)
            ]
            # [
            #     random.sample(image_list, random.randint(1, len(image_list)))
            #     for _ in range(random.randint(1, len(image_list)))
            # ]
        )
        mixed_data = mixed_data.append(row_i, ignore_index=True)

    return mixed_data


def save_images_to_disk(data, image_dir):
    # Create the image directory if it doesn't exist
    os.makedirs(image_dir, exist_ok=True)

    # Save the images to disk and replace the image column with the image paths
    for i, row in data.iterrows():
        img = row[IMAGE_COLUMN]
        img_path = os.path.join(image_dir, f"{i}.jpg")
        img.save(img_path)
        data.at[i, IMAGE_COLUMN] = img_path
