import os
import random
import numpy as np

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
        )
        image = row_i[IMAGE_COLUMN]
        sqrWidth = np.ceil(np.sqrt(image.size[0] * image.size[1])).astype(int)
        im_resize = image.resize((sqrWidth, sqrWidth))
        row_i[IMAGE_COLUMN] = im_resize
        mixed_data = mixed_data.append(row_i, ignore_index=True)

    return mixed_data


def get_flatten(group):
    attributes_set = set()
    for item in group:
        if isinstance(item, list):
            for item2 in item:
                attributes_set.add(item2)
        else:
            attributes_set.add(item)
    return attributes_set


def get_balanced_by_gender_race(df: pd.DataFrame, split_seed: int):
    groups = list(zip(df[GENDER_COLUMN], df[RACE_COLUMN]))

    flatten_groups = []
    for group in groups:
        flatten_groups.append(get_flatten(group))

    used_groups = []
    groups_counter = []
    for group in flatten_groups:
        if group in used_groups:
            for i in range(len(groups_counter)):
                if groups_counter[i][0] == group:
                    groups_counter[i][1] += 1
                    break
        else:
            used_groups.append(group)
            groups_counter.append([group, 1])

    min_count = min([x[1] for x in groups_counter])

    balanced_data = pd.DataFrame(columns=df.columns)

    for gender, race in list(zip(df[GENDER_COLUMN], df[RACE_COLUMN])):
        flatten_gender = (
            get_flatten(gender) if isinstance(gender, list) else set([gender])
        )
        flatten_race = get_flatten(race) if isinstance(race, list) else set([race])

        used_group = flatten_gender.union(flatten_race)
        if used_group not in used_groups:
            continue
        used_groups.remove(used_group)
        group = df[
            (df[GENDER_COLUMN].apply(lambda x: x == gender))
            & (df[RACE_COLUMN].apply(lambda x: x == race))
        ]
        sample = (
            group.sample(n=min_count, random_state=split_seed)
            if len(group) > min_count
            else group
        )
        balanced_data = pd.concat([balanced_data, sample])

    balanced_data = balanced_data.sample(frac=1, random_state=split_seed)

    return balanced_data


def save_images_to_disk(data, image_dir):
    # Create the image directory if it doesn't exist
    os.makedirs(image_dir, exist_ok=True)

    # Save the images to disk and replace the image column with the image paths
    for i, row in data.iterrows():
        img = row[IMAGE_COLUMN]
        img_path = os.path.join(image_dir, f"{i}.jpg")
        img.save(img_path)
        data.at[i, IMAGE_COLUMN] = img_path
