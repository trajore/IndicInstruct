import argparse
from datasets import load_dataset
import numpy as np

def main(args):
    # Load dataset
    dataset = load_dataset("Divyanshu/indicxnli", f"{args.lang}")
    
    # Map premise and hypothesis to strip leading and trailing whitespaces
    dataset = dataset.map(lambda x: {"premise": x["premise"].strip()})
    dataset = dataset.map(lambda x: {"hypothesis": x["hypothesis"].strip()})
    
    # Get validation and test data
    dev_data = dataset["validation"]
    test_data = dataset["test"]
    
    # Get ground truth labels for validation and test data
    dev_labels = np.array(dev_data["label"])
    test_labels = np.array(test_data["label"])
    
    # Save ground truth labels to files
    np.savetxt(args.dev_labels_file, dev_labels, fmt="%d")
    np.savetxt(args.test_labels_file, test_labels, fmt="%d")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="hi", choices=["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "te"])
    parser.add_argument("--dev_labels_file", type=str, default="dev_labels.txt", help="Path to save the ground truth labels for validation data.")
    parser.add_argument("--test_labels_file", type=str, default="test_labels.txt", help="Path to save the ground truth labels for test data.")
    args = parser.parse_args()
    main(args)


