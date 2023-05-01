import fasttext
import argparse
from faq50 import FAQ


parser = argparse.ArgumentParser()
parser.add_argument("model_path", default="")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    model = fasttext.load_model(args.model_path)

    faq = FAQ(model, "Q50_questions.xlsx", "Q50_answers.xlsx")
    faq.total_confusion()

