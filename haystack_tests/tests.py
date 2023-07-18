import argparse
import pandas as pd
from retriever import FastTextRetriever, extract_word_probs
from haystack.document_stores import FAISSDocumentStore


parser = argparse.ArgumentParser()
parser.add_argument("model_path", default="")
parser.add_argument("--probs", default="", help="Word probabilities file path")
parser.add_argument("--alpha", default=1e-4, type=float, help="Word embedding weighting factor")
parser.add_argument("--compressed", default=False, action="store_true", help="Indicate if the used model was compressed using compress-fasttext")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    if args.probs == "":
        args.probs = None

    def test_question_set(path):
        document_store = FAISSDocumentStore(sql_url="sqlite:///", embedding_dim=300)

        df = pd.read_excel(path)
        df = df.rename(columns={'question': 'content'})
        dicts = df.to_dict('records')
        document_store.write_documents(dicts)

        retriever = FastTextRetriever(
            document_store, 
            args.model_path,
            alpha=args.alpha,
            w_probs=args.probs,
            compressed=args.compressed
        )

        hits = 0
        weird = 0
        for idx in df.index:
            q = df['content'][idx]
            cls = df['class'][idx]
            docs = retriever.retrieve(q)
            second = docs[1].to_dict()
            if q != docs[0].to_dict()['content']:
                weird += 1
                print("Ambiguous match:")
                print(f'{q}, cls {cls}')
                print(f"{docs[0].to_dict()['content']}, cls {docs[0].to_dict()['meta']['class']}")
                print()
            if cls == second['meta']['class']:
                hits += 1
        acc = 1.0 * hits / len(df.index)
        #print(f'TOP ONE FAILS: {weird}/{len(df.index)}')
        return acc
    
    acc50 = test_question_set("../data/FAQ50_questions.xlsx")
    acc76 = test_question_set("../data/FAQ76_questions.xlsx")
    acc79 = test_question_set("../data/FAQ79_questions.xlsx")
    print(f'UPV50 Cross-match ACC: {acc50}')
    print(f'UPV76 Cross-match ACC: {acc76}')
    print(f'UPV79 Cross-match ACC: {acc79}')

    