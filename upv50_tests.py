import fasttext
import os
import argparse
from faq50 import FAQ


parser = argparse.ArgumentParser()
parser.add_argument("model_path", default="")
parser.add_argument("--verb", default=False, action="store_true", help="Print incorrect matches")
parser.add_argument("--cm", default=False, action="store_true", help="Create and show a confusion matrix")
parser.add_argument("--save", default=False, action="store_true", help="Save results")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    
    model = fasttext.load_model(args.model_path)

    faq = FAQ(model, "Q50_questions.xlsx", "Q50_answers.xlsx")
    q_acc, q_cm = faq.cross_match_test(verb=args.verb, show_cm=args.cm)
    a_acc, a_cm = faq.ans_test(verb=args.verb, show_cm=args.cm)
    #m_acc, m_cm = faq.mean_match_test(verb=args.verb, show_cm=args.cm)
    #faq.total_confusion()

    if args.save:
        save_dir = args.model_path.replace(".bin", "_STS")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "Accuracies.log"), "w") as af:
            af.writelines([f"Question matching accuracy: {q_acc} \nAnswer matching accuracy: {a_acc}"])
        if args.cm:
            q_cm.savefig(os.path.join(save_dir, "Question_CM.png"))
            a_cm.savefig(os.path.join(save_dir, "Answer_CM.png"))
