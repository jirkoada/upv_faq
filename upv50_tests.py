import fasttext
import os
import argparse
import json
from faq_core import FAQ


parser = argparse.ArgumentParser()
parser.add_argument("model_path", default="")
parser.add_argument("--probs", default="", help="Word probabilities file path")
parser.add_argument("--alpha", default=1e-4, type=float, help="Word embedding weighting factor")
parser.add_argument("--verb", default=False, action="store_true", help="Print incorrect matches")
parser.add_argument("--cm", default=False, action="store_true", help="Create and show a confusion matrix")
parser.add_argument("--cmtime", default=0.1, type=float, help="Confusion matrix display duration")
parser.add_argument("--save", default=False, action="store_true", help="Save results")
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

    faq = FAQ(model, "Q50_questions.csv", "Q50_answers.csv", probs=probs, alpha=args.alpha, compressed=args.compressed)
    q_acc, q_cm = faq.cross_match_test(verb=args.verb, show_cm=args.cm, show_time=args.cmtime)
    a_acc, a_cm = faq.ans_test(verb=args.verb, show_cm=args.cm, show_time=args.cmtime) 

    print(f"Question cross-match accuracy: {q_acc}")
    print(f"Answer match accuracy: {a_acc}")

    if args.save:
        save_dir = args.model_path.replace(".bin", f"_STS50_alpha={args.alpha}")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "Accuracies.log"), "w") as af:
            af.writelines([f"Question matching accuracy: {q_acc} \nAnswer matching accuracy: {a_acc}"])
        if args.cm:
            q_cm.savefig(os.path.join(save_dir, "Question_CM.png"))
            a_cm.savefig(os.path.join(save_dir, "Answer_CM.png"))

    '''
    print("Weighted")
    faq = FAQ(model, "Q50_questions.xlsx", "Q50_answers.xlsx", alpha=0.0001, compressed=args.compressed, probs_path=args.probs)
    q_acc, q_cm = faq.cross_match_test(verb=args.verb, show_cm=args.cm)
    a_acc, a_cm = faq.ans_test(verb=args.verb, show_cm=args.cm)

    print("Basic")
    faq = FAQ(model, "Q50_questions.xlsx", "Q50_answers.xlsx", compressed=args.compressed)
    q_acc, q_cm = faq.cross_match_test(verb=args.verb, show_cm=args.cm)
    a_acc, a_cm = faq.ans_test(verb=args.verb, show_cm=args.cm)
    '''
    
