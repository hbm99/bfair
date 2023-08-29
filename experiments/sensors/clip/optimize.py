
import argparse
import traceback
from pathlib import Path

import pandas as pd

from bfair.datasets import load_utkface
from bfair.datasets.utkface import (GENDER_COLUMN, GENDER_VALUES, RACE_COLUMN,
                                    RACE_VALUES, RACE_VALUES_WITH_NOT_RULE)
from bfair.sensors import P_GENDER, P_RACE
from bfair.sensors.image.clip.optimization import optimize
from bfair.sensors.optimization import (MACRO_ACC, MACRO_F1, MACRO_PRECISION,
                                        MACRO_RECALL, MICRO_ACC)

DB_UTKFACE = 'utkface'

SENSOR_CLIPBASED = "clipbased"

IMAGE_COLUMN = 'image'

def setup():
    parser = argparse.ArgumentParser()

    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--eval-timeout", type=int, default=1000)
    parser.add_argument("--memory", type=int, default=1000)
    parser.add_argument("--popsize", type=int, default=50)
    parser.add_argument("--global-timeout", type=int, default=60)
    parser.add_argument("--token", default=None)
    parser.add_argument("--channel", default=None)
    parser.add_argument("--output", default="")
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--metric",
        action="append",
        choices=[MACRO_F1, MACRO_PRECISION, MACRO_RECALL, MACRO_ACC, MICRO_ACC],
        default=[],
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=[DB_UTKFACE],
        default=[DB_UTKFACE],
    )
    parser.add_argument(
        "--skip",
        action="append",
        choices=[SENSOR_CLIPBASED],
        default=[],
    )
    parser.add_argument(
        "--force",
        action="append",
        choices=[
            SENSOR_CLIPBASED
        ],
        default=[],
    )

    return parser.parse_args()


def main():
    args = setup()

    if args.output:
        Path(args.output).parent.mkdir(exist_ok=True)
        output_stream = open(args.output, mode="a")
    else:
        output_stream = None

    try:
        images_for_training = []
        annotations_for_training = []
        images_for_testing = []
        annotations_for_testing = []
        attr_cls = P_GENDER
        sensitive_values = ["Male", "Female"]

        if DB_UTKFACE in args.dataset:
            dataset = load_utkface(split_seed=0)
            images_for_training.append(dataset.data[IMAGE_COLUMN])
            annotations_for_training.append(dataset.data[GENDER_COLUMN])
            images_for_testing.append(dataset.test[IMAGE_COLUMN])
            annotations_for_testing.append(dataset.test[GENDER_COLUMN])


        X_train = pd.concat(images_for_training)
        y_train = pd.concat(annotations_for_training)
        X_test = pd.concat(images_for_testing)
        y_test = pd.concat(annotations_for_testing)

        best_solution, best_fn, search = optimize(
            X_train,
            y_train,
            X_test,
            y_test,
            sensitive_values,
            attr_cls,
            score_key=args.metric if args.metric else [MACRO_F1],
            force_clip_based_sensor=True,
            pop_size=args.popsize,
            search_iterations=args.iterations,
            evaluation_timeout=args.eval_timeout,
            memory_limit=args.memory * 1024**3 if args.memory else None,
            search_timeout=args.global_timeout,
            errors="warn",
            inspect=True,
            output_stream=output_stream,
        )

        print("Best solution", file=output_stream)
        print(best_fn, file=output_stream)
        print(best_solution, file=output_stream, flush=True)

    except Exception as e:
        print(
            "\n",
            "ERROR",
            "\n",
            str(e),
            "\n",
            traceback.format_exc(),
            "\n",
            file=output_stream,
            flush=True,
        )
    finally:
        if output_stream is not None:
            output_stream.close()


if __name__ == "__main__":
    main()
