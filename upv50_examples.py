import fasttext
import json
import os
from faq_core import FAQ, extract_word_probs

model_path = "models/cbow_300_ns10_800k_ep10.bin"
probs_path = "cbow_300_ns10_800k_ep10_probs.json"
compressed_model = False

extract_word_probs(model_path, corpus_size=13.2e9) # run only once to save time

with open(probs_path, "r") as wp_file:
    probs = json.load(wp_file)

if compressed_model:
    import compress_fasttext
    model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(model_path)
else:
    model = fasttext.load_model(model_path)

faq = FAQ(model, "Q50_questions.xlsx", "Q50_answers.xlsx", probs=probs, compressed=compressed_model)

test_question = "Jak požádat o patent?"
matched = faq.match(test_question)
answer = faq.answer(test_question)
direct_answer = faq.direct_answer(test_question)

print(f"\nQuestion:\n{test_question}")
print(f"\nBest Match:\n{matched}")
print(f"\nBest Match Answer:\n{answer}")
print(f"\nBest Directly Matched Answer:\n{direct_answer}")

acc, cm = faq.cross_match_test()
print(f"\nQuestion Cross-Match Accuracy: {acc}")