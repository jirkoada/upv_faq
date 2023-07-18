import pandas as pd
import json


def remove_tags():
    ans_df = pd.read_excel("FAQ79_answers.xlsx")
    ans_df["answer"] = ans_df["answer"].str.replace(r'<sub alias="([^"]*)">[^<]*</sub>', r'\1', regex=True)
    ans_df.to_excel("FAQ79_answers_no_tags.xlsx")


def count_word_probs():
    ans_df = pd.read_excel("FAQ79_answers_no_tags.xlsx")
    corpus = ans_df['answer'].str.cat(sep="\n")
    words = corpus.lower().split()
    #print(words)
    #print(len(words))
    
    probs = {}
    for word in words:
        word = word.strip('.')
        if word in probs.keys():
            probs[word] += 1
        else:
            probs[word] = 1
    for key in probs.keys():
        probs[key] /= len(words)
    #print(probs)
    with open("FAQ79_answer_word_probs.json", "w") as f:
        json.dump(probs, f)


if __name__ == "__main__":
    #remove_tags()
    count_word_probs()
