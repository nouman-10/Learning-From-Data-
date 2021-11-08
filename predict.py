import argparse
import os
import pickle

from model import BaselineModel, identity


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--data_dir",
        default="./data/",
        type=str,
        help="Input directory to read json files from (default ./data/)",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default="./output/",
        type=str,
        help="Output directory to save the results to (default ./output/)",
    )
    parser.add_argument(
        "-m",
        "--saved_model_path",
        default="./data/best_baseline_model.pkl",
        type=str,
        help="Read saved model from the path (default ./data/best_baseline_model.pkl)",
    )
    parser.add_argument(
        "-t", "--test_path", default="test.json", type=str, help="Test json file from (default test.json)",
    )
    args = parser.parse_args()
    return args


def load_model(path):
    """Load model from the given path"""
    with open(path, "rb") as f:
        return pickle.load(f)


def read_labels(path):
    """Read labels from the given file"""
    with open(path, "r") as f:
        return f.read().split("\n")


if __name__ == "__main__":
    args = create_arg_parser()

    # Test the best saved model on the test file
    model = BaselineModel(
        data_path=args.data_dir,
        model_type="svm",
        model_params={"kernel": "linear"},
        vectorizer_type="tfidf",
        vectorizer_params={"max_df": 0.95, "max_features": 50000, "ngram_range": (1, 2)},
        extra_feature=None,
    )
    model.labels = read_labels(os.path.join(args.data_dir, "labels.txt"))
    best_model = load_model(args.saved_model_path)
    model.pipeline = best_model
    model.read_data(only_test=True, test_path=args.test_path)
    model.evaluate()
    model.print_results()
    model.save_results(os.path.join(args.output_path, "test_results.txt"))
