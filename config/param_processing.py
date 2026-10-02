import argparse
import json
import os

import pandas as pd


def load_args():

    # Create parser
    parser = argparse.ArgumentParser(
        description="Read class parameters from VisualAtom-1k"
    )
    # Add path argument
    parser.add_argument(
        "param-folder", type=str, help="Path to param folder of VisualAtom-1k"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        help="Path to store parameters of VisualAtom-1k as a single csv",
        default="classes.csv",
    )
    # Parse args and return
    args = parser.parse_args()
    return args


def main():

    # Load args
    args = load_args()

    # Create empty lists
    classes = {}

    # Read all csv files inside the folder
    for file in sorted(os.listdir(args.param_folder)):

        # Read csv
        df = pd.read_csv(
            os.path.join(args.param_folder, file),
            header=None,
            names=["key", "value"],
        )

        # Read as numeric
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # Process csv
        for key, value in zip(df["key"], df["value"]):
            classes.setdefault(key, []).append(value)

    # Convert to csv
    classes_df = pd.DataFrame(classes)
    classes_df.to_csv(args.save_path, index=False)


if __name__ == "__main__":
    main()
