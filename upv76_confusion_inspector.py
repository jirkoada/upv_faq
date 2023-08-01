import fasttext
import argparse
import json
from faq_core import FAQ


parser = argparse.ArgumentParser()
parser.add_argument("model_path", default="")
parser.add_argument("--probs", default="", help="Word probabilities file path")
parser.add_argument("--alpha", default=1e-4, type=float, help="Word embedding weighting factor")
parser.add_argument("--compressed", default=False, action="store_true", help="Indicate if the used model was compressed using compress-fasttext")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    if args.compressed:
        import compress_fasttext
        model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(args.model_path)
    else:
        model = fasttext.load_model(args.model_path)

    probs = None
    if args.probs == "":
        args.alpha = None
    else:
        with open(args.probs, "r") as wp_file:
            probs = json.load(wp_file)

    faq = FAQ(model, "data/FAQ76v2_questions.csv", "data/FAQ76v2_answers.csv", probs=probs, alpha=args.alpha, compressed=args.compressed)
    faq.total_confusion()

