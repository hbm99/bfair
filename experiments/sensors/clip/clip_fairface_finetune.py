# -*- coding: utf-8 -*-
"""clip_fairface_finetune.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16Dbl5PkTUt-VZCsX5fOMcLZAOeALmqT-

Loading dataset
"""

# Commented out IPython magic to ensure Python compatibility.

from bfair.datasets.build_tools.fairface import (
    create_balanced_dataset,
    create_mixed_dataset,
)
from statistics import mean
import datasets as db
import pandas as pd
import numpy as np

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
CIFAR_IMAGE_COLUMN = "img"

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

SIZE = 3000
BALANCED = True

"""utils"""

split_seed = 0

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

if BALANCED:
    mixed_data = create_balanced_dataset(new_df_noisy, SIZE, 0)
else:
    mixed_data = create_mixed_dataset(new_df_noisy, SIZE, 0)

"""Loaded dataset!

Loading CLIP model
"""

# Commented out IPython magic to ensure Python compatibility.

import torch
import clip


BATCH_SIZE = 1
EPOCH = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

from torch.utils.data import Dataset, DataLoader

train_df_temp = mixed_data.copy()

train_df_temp[GENDER_COLUMN] = train_df_temp[GENDER_COLUMN].apply(
    lambda x: sorted(x) if isinstance(x, list) else [x]
)
train_df_temp[RACE_COLUMN] = train_df_temp[RACE_COLUMN].apply(
    lambda x: sorted(x) if isinstance(x, list) else [x]
)

train_df_temp[GENDER_COLUMN] = train_df_temp[GENDER_COLUMN].apply(lambda x: repr(x))
train_df_temp[RACE_COLUMN] = train_df_temp[RACE_COLUMN].apply(lambda x: repr(x))

train_df_temp = train_df_temp.groupby([GENDER_COLUMN, RACE_COLUMN]).sample(frac=0.8)

import ast

train_df_temp[GENDER_COLUMN] = train_df_temp[GENDER_COLUMN].apply(
    lambda x: ast.literal_eval(x)
)
train_df_temp[RACE_COLUMN] = train_df_temp[RACE_COLUMN].apply(
    lambda x: ast.literal_eval(x)
)

train_df_temp[GENDER_COLUMN] = train_df_temp[GENDER_COLUMN].apply(
    lambda x: x[0] if len(x) == 1 else x
)
train_df_temp[RACE_COLUMN] = train_df_temp[RACE_COLUMN].apply(
    lambda x: x[0] if len(x) == 1 else x
)


validation_df = mixed_data[~mixed_data[IMAGE_COLUMN].isin(train_df_temp[IMAGE_COLUMN])]
validation_df = validation_df.reset_index(drop=True)
train_df = train_df_temp.reset_index(drop=True)


class NoisyFFDataset(Dataset):
    def __init__(self, dataframe, preprocess):
        self.preprocess = preprocess
        self.image = dataframe["image"].tolist()
        self.gender = dataframe["gender"].tolist()
        self.race = dataframe["race"].tolist()
        self.age = dataframe["age"].tolist()

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = self.preprocess(self.image[idx])
        gender = self.gender[idx]
        race = self.race[idx]
        age = self.age[idx]
        return image, gender, race, age


train_dataset = NoisyFFDataset(train_df, preprocess)
validation_dataset = NoisyFFDataset(validation_df, preprocess)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(
    validation_dataset, batch_size=BATCH_SIZE, shuffle=False
)


attributes_queries = {}

for label in RACE_VALUES:
    attributes_queries[label] = "A photo of a " + label.lower() + " race person."

for label in GENDER_VALUES:
    attributes_queries[label] = "A photo of a " + label.lower() + " gender person."

# print(attributes_queries)

gender_texts = [attributes_queries[lbl] for lbl in GENDER_VALUES]
# print(gender_texts)
gender_texts = clip.tokenize(gender_texts).to(device)


race_texts = [attributes_queries[lbl] for lbl in RACE_VALUES]
# print(race_texts)
race_texts = clip.tokenize(race_texts).to(device)

from torch import nn, optim
from tqdm import tqdm
import numpy as np


# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


def safe_division(numerator, denominator, default=0):
    return numerator / denominator if denominator else default


def print_scores(column_name, scores):
    print(f"{column_name} scores:")
    for key, value in scores.items():
        print(f"{key}: {value}")


def compute_scores(ir_counter, ac_counter):
    scores = {}
    per_group = []
    for value, (correct_hit, spurious, missing, _) in ir_counter.items():
        precision = safe_division(correct_hit, correct_hit + spurious)
        recall = safe_division(correct_hit, correct_hit + missing)
        f1 = safe_division(2 * precision * recall, precision + recall)
        scores[value] = group = {
            PRECISION: precision,
            RECALL: recall,
            F1: f1,
        }
        per_group.append(group)

    scores[MACRO_PRECISION] = mean(group[PRECISION] for group in per_group)
    scores[MACRO_RECALL] = mean(group[RECALL] for group in per_group)
    scores[MACRO_F1] = mean(group[F1] for group in per_group)

    total_correct = 0
    total_total = 0
    total_accuracy = 0
    for value, (correct, total) in ac_counter.items():
        total_correct += correct
        total_total += total
        total_accuracy += safe_division(correct, total)

    scores[MICRO_ACC] = safe_division(total_correct, total_total)
    scores[MACRO_ACC] = safe_division(total_accuracy, len(ac_counter))

    return scores


def compute_errors(values, ir_counter, ac_counter, true_ann, pred_ann):
    for i in range(len(values)):
        if values[i] not in ir_counter.keys():
            ir_counter[values[i]] = (0, 0, 0, 0)
        correct_hit, spurious, missing, correct_rejection = ir_counter[values[i]]
        if true_ann[i] == 1 and pred_ann[i] == 0:
            missing += 1
        elif pred_ann[i] == 1 and true_ann[i] == 0:
            spurious += 1
        elif true_ann[i] == 1:
            correct_hit += 1
        else:
            correct_rejection += 1
        ir_counter[values[i]] = (
            correct_hit,
            spurious,
            missing,
            correct_rejection,
        )

    true_ann_key = ""
    for item in true_ann:
        true_ann_key += str(int(item))

    correct, total = ac_counter.get(true_ann_key, (0, 0))
    equal = int(true_ann == pred_ann)
    ac_counter[true_ann_key] = (correct + equal, total + 1)


def set_ground_truth(values, place, batch, ground_truth, i):
    truth_idxs = []
    for item in batch[place]:
        if isinstance(item, tuple):
            item = item[0]
        if item in values:
            truth_idxs.append(values.index(item))
    for truth_idx in truth_idxs:
        ground_truth[i, truth_idx] = 1


def get_pred(logits_per_image):
    batch_probs = torch.sigmoid(logits_per_image).to(device).numpy()
    rounded_batch_probs = np.round(batch_probs)
    return rounded_batch_probs.tolist()[0]


if device == "cpu":
    model.float()

loss_img = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, len(train_dataloader) * EPOCH
)

PRECISION = "precision"
RECALL = "recall"
F1 = "f1"
MACRO_PRECISION = "macro-precision"
MACRO_RECALL = "macro-recall"
MACRO_F1 = "macro-f1"

MICRO_ACC = "micro-accuracy"
MACRO_ACC = "macro-accuracy"

best_gender_mac_acc = -1
best_gender_ep = -1
GENDER_PLACE = 1

best_race_mac_acc = -1
best_race_ep = -1
RACE_PLACE = 2


for epoch in range(EPOCH):
    print(
        f"running epoch {epoch}, GENDER: best test macro-accuracy {best_gender_mac_acc} after epoch {best_gender_ep}, RACE: best test macro-accuracy {best_race_mac_acc} after epoch {best_race_ep}"
    )
    step = 0
    gender_tr_loss = 0
    race_tr_loss = 0
    model.train()
    pbar = tqdm(train_dataloader, leave=False)
    for batch in pbar:
        # print(batch)
        step += 1
        optimizer.zero_grad()

        images = batch[0]

        images = images.to(device)

        gender_logits_per_image, _ = model(images, gender_texts)
        race_logits_per_image, _ = model(images, race_texts)

        gender_ground_truth = torch.zeros((BATCH_SIZE, len(GENDER_VALUES))).to(device)
        race_ground_truth = torch.zeros((BATCH_SIZE, len(RACE_VALUES))).to(device)

        for i in range(BATCH_SIZE):
            # gender
            gender_truth_idxs = []
            for item in batch[GENDER_PLACE]:
                if isinstance(item, tuple):
                    item = item[0]
                if item in GENDER_VALUES:
                    gender_truth_idxs.append(GENDER_VALUES.index(item))
            for gender_truth_idx in gender_truth_idxs:
                gender_ground_truth[i, gender_truth_idx] = 1

            # race
            race_truth_idxs = []
            for item in batch[RACE_PLACE]:
                if isinstance(item, tuple):
                    item = item[0]
                if item in RACE_VALUES:
                    race_truth_idxs.append(RACE_VALUES.index(item))
            for race_truth_idx in race_truth_idxs:
                race_ground_truth[i, race_truth_idx] = 1

        gender_total_loss = loss_img(gender_logits_per_image, gender_ground_truth)
        gender_total_loss.backward()
        gender_tr_loss += gender_total_loss.item()

        race_total_loss = loss_img(race_logits_per_image, race_ground_truth)
        race_total_loss.backward()
        race_tr_loss += race_total_loss.item()

        if device == "cpu":
            optimizer.step()
            scheduler.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            scheduler.step()
            clip.model.convert_weights(model)

        pbar.set_description(
            f"GENDER: train batchCE: {gender_total_loss.item()}, RACE: train batchCE: {race_total_loss.item()}",
            refresh=True,
        )

    gender_tr_loss /= step
    race_tr_loss /= step

    gender_ir_counter = {}
    gender_ac_counter = {}
    race_ir_counter = {}
    race_ac_counter = {}
    step = 0
    gender_te_loss = 0
    race_te_loss = 0
    with torch.no_grad():
        model.eval()
        val_pbar = tqdm(validation_dataloader, leave=False)
        for batch in val_pbar:
            step += 1
            images = batch[0]

            images = images.to(device)

            gender_ground_truth = torch.zeros((BATCH_SIZE, len(GENDER_VALUES))).to(
                device
            )

            race_ground_truth = torch.zeros((BATCH_SIZE, len(RACE_VALUES))).to(device)

            for i in range(BATCH_SIZE):
                set_ground_truth(
                    GENDER_VALUES, GENDER_PLACE, batch, gender_ground_truth, i
                )

                set_ground_truth(RACE_VALUES, RACE_PLACE, batch, race_ground_truth, i)

            gender_total_loss = loss_img(gender_logits_per_image, gender_ground_truth)
            gender_te_loss += gender_total_loss.item()

            race_total_loss = loss_img(race_logits_per_image, race_ground_truth)
            race_te_loss += race_total_loss.item()

            gender_ground_truth = gender_ground_truth.tolist()[0]
            race_ground_truth = race_ground_truth.tolist()[0]

            gender_logits_per_image, _ = model(images, gender_texts)
            rounded_g_b_p = get_pred(gender_logits_per_image)

            race_logits_per_image, _ = model(images, race_texts)
            rounded_r_b_p = get_pred(race_logits_per_image)

            # Compute gender errors
            compute_errors(
                GENDER_VALUES,
                gender_ir_counter,
                gender_ac_counter,
                gender_ground_truth,
                rounded_g_b_p,
            )

            # Compute race errors
            compute_errors(
                RACE_VALUES,
                race_ir_counter,
                race_ac_counter,
                race_ground_truth,
                rounded_r_b_p,
            )

            val_pbar.set_description(
                f"gender test batchCE: {gender_total_loss.item()}, race test batchCE: {race_total_loss.item()}",
                refresh=True,
            )

        gender_te_loss /= step
        race_te_loss /= step

    # Compute gender scores
    gender_scores = compute_scores(gender_ir_counter, gender_ac_counter)

    # Print gender scores
    print_scores(GENDER_COLUMN, gender_scores)

    print()

    # Compute race scores
    race_scores = compute_scores(race_ir_counter, race_ac_counter)

    # Print race scores
    print_scores(RACE_COLUMN, race_scores)

    # Model selection
    gender_macro_acc = gender_scores[MACRO_ACC]
    race_macro_acc = race_scores[MACRO_ACC]
    if gender_macro_acc >= best_gender_mac_acc and race_macro_acc >= best_race_mac_acc:
        best_gender_mac_acc = gender_macro_acc
        best_race_mac_acc = race_macro_acc

        best_gender_ep = epoch
        best_race_ep = epoch

        best_model = model
    print(
        f"epoch {epoch}, gender_tr_loss {gender_tr_loss}, gender_te_loss {gender_te_loss}, gender_macro_acc {gender_macro_acc}, race_tr_loss {race_tr_loss}, race_te_loss {race_te_loss}, race_macro_acc {race_macro_acc}"
    )

torch.save(best_model.cpu().state_dict(), "best_model.pt")
