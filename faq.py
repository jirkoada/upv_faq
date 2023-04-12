import fasttext
import numpy as np
import warnings
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt


class FAQ:
    def __init__(self, model, questions, answers=None):
        self.model = model
        self.questions = questions
        self.answers = answers
        self.db = np.array([self.sentence_embedding(q) for q in questions])
        if not answers is None:
            assert len(questions) == len(answers)
            self.ans_db = np.array([self.sentence_embedding(a) for a in answers])
    
    def sentence_embedding(self, sentence):
        #return self.model.get_sentence_vector(sentence.replace('\n', ' '))  # Doesn't work well
        word_list = sentence.lower().split()
        #word_list = sentence.split()
        for idx, word in enumerate(word_list):
            word_list[idx] = word.strip(".,!?:")
        assert(len(word_list) > 0)
        embedding = np.array([self.model.get_word_vector(x) for x in word_list]).mean(0)
        return embedding/np.linalg.norm(embedding)
    
    def identify(self, question):
        v = self.sentence_embedding(question)
        sims = self.db @ v[:, np.newaxis]
        return np.argmax(sims)
    
    def identify_direct_answer(self, question):
        v = self.sentence_embedding(question)
        a_sims = self.ans_db @ v[:, np.newaxis]
        return np.argmax(a_sims)
    
    def match(self, question):
        matched_q = self.questions[self.identify(question)]
        print(f"Matched question: {matched_q}")
        return matched_q

    def answer(self, question):
        if not self.answers:
            warnings.warn("Answers are not available")
            return None
        ans = self.answers[self.identify(question)]
        print(f"Answer: {ans}")
        return ans
    
    def direct_answer(self, question):
        if not self.answers:
            warnings.warn("Answers are not available")
            return None
        ans = self.answers[self.identify_direct_answer(question)]
        print(f"Answer: {ans}")
        return ans

    def test(self, questions, ids, verb=False, show_cm=False):
        assert(len(questions) == len(ids))
        hyps = np.array([self.identify(q) for q in questions])
        ids = np.array(ids)
        scores = hyps == ids
        acc = scores.mean()
        print(f"Question match accuracy: {acc}")
        if not scores.all() and verb:
            print("\nIncorrect matches:")
            for i, b in enumerate(scores):
                if not b:
                    print(f"{questions[i]} : {self.questions[hyps[i]]}")
        if show_cm:
            cm = confusion_matrix(ids, hyps)
            fig = plt.figure(figsize = (10,7))
            sn.heatmap(cm, annot=True)
            plt.title("Question matching confusion matrix")
            plt.draw()
            plt.pause(0.1)
            return acc, fig
        return acc, None

    def ans_test(self, questions, ids, verb=False, show_cm=False):
        assert(len(questions) == len(ids))
        if not self.answers:
            warnings.warn("Answers are not available")
            return None
        hyps = np.array([self.identify_direct_answer(q) for q in questions])
        ids = np.array(ids)
        scores = hyps == ids
        acc = scores.mean()
        print(f"Answer match accuracy: {acc}")
        if not scores.all() and verb:
            print("\nIncorrect matches:")
            for i, b in enumerate(scores):
                if not b:
                    print(f"{questions[i]} : {self.answers[hyps[i]]}")
        if show_cm:
            cm = confusion_matrix(ids, hyps)
            fig = plt.figure(figsize = (10,7))
            sn.heatmap(cm, annot=True)
            plt.title("Answer matching confusion matrix")
            plt.draw()
            plt.pause(0.1)
            return acc, fig
        return acc, None
