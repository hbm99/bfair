from functools import partial

from autogoal import datasets
from autogoal.datasets import (
    abalone,
    cars,
    dorothea,
    german_credit,
    gisette,
    shuttle,
    yeast,
)
from autogoal.kb import MatrixContinuousDense
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from experiments.core import run, setup

valid_datasets = [
    "abalone",
    "cars",
    "dorothea",
    "gisette",
    "shuttle",
    "yeast",
    "german_credit",
]


def load_dataset(name, max_examples=None):
    data = getattr(datasets, name).load()
    if len(data) == 4:
        X_train, y_train, X_test, y_test = data
    else:
        X, y = data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return (
        X_train[:max_examples],
        y_train[:max_examples],
        X_test[:max_examples],
        y_test[:max_examples],
    )


def main():
    args = setup()
    selected_datasets = [args.title] if args.title in valid_datasets else valid_datasets
    for name in selected_datasets:
        run(
            load_dataset=partial(load_dataset, name=name),
            input_type=MatrixContinuousDense,
            score_metric=accuracy_score,
            maximize=True,
            args=args,
            title=name,
        )


if __name__ == "__main__":
    main()
