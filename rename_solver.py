import os
import argparse
import sys

from utils import load_results
import pickle

HERE_TEST_PATH = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(HERE_TEST_PATH, './genosolver/')):
    sys.path.insert(0, os.path.join(HERE_TEST_PATH, './genosolver/'))
    import genosolver
    sys.path.pop(0)
else:
    raise ImportError('No genosolver folder')

if os.path.exists(os.path.join(HERE_TEST_PATH, "./bayesian-geno/")):
    sys.path.insert(0, os.path.join(HERE_TEST_PATH, "./bayesian-geno/"))

    import bayesian_line_search.GPgenosolver

    sys.path.pop(0)
else:
    print(HERE_TEST_PATH)
    raise ImportError("No bayesian genosolver folder")

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
