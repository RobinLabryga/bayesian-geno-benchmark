import os
import argparse

from utils import load_results
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pickle helper.", add_help=False)
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="For loading .pkl results of a run.",
    )
    parser.add_argument("-f", "--file", type=str, help="Result to load.")
    parser.add_argument("-i", "--in_name", type=str, help="Original name of solver.")
    parser.add_argument("-o", "--out_name", type=str, help="New name of solver.")
    args = parser.parse_args()

    results = load_results(args.file)

    for problem, result in results.items():
        if args.in_name in result:
            result[args.out_name] = result[args.in_name]
            result.pop(args.in_name)

    assert os.path.exists(args.file), f"Result file {args.file} doesn't exist"
    with open(args.file, "wb") as f:
        pickle.dump(results, f)
