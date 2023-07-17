import haystack
from haystack import Document
from haystack.document_stores import BaseDocumentStore, FAISSDocumentStore
from typing import *
import numpy as np
import fasttext
import json
import os
#from nltk.tokenize import word_tokenize

def extract_word_probs(model_path: str, corpus_size: int = 4.1e9):
    """
    Uses word occurency counts included in FT model and training corpus size to create a dict of 
    word frequencies/probabilities and saves it to a json file inside current working directory
    """
    model = fasttext.load_model(model_path)
    words, freqs = model.get_words(include_freq=True)
    probs = list(map(lambda x: float(x) / corpus_size, freqs))
    word_probs = dict(zip(words, probs))
    with open(os.path.splitext(os.path.basename(model_path))[0] + "_probs.json", "w") as out:
        json.dump(word_probs, out)
    

class FastTextRetriever(haystack.nodes.retriever.BaseRetriever):
    """
    Changes made:
    -   Changed parent class to BaseRetriever, as DenseRetriever was missing in my haystack==1.5.0 install
    -   Added an optional alternative text embedding algorithm according to: https://openreview.net/pdf?id=SyK00v5xx
            - uses word probabilities and a tunable parameter alpha to weigh word embeddings before averaging them
            - higher significance for less common words
    -   Added support for compressed fastText models
            - The required library compress_fasttext is imported and used only if the parameter compressed is set to True
    -   Added 3 parameters:
            alpha - can be tuned for best matching performance, 1e-4 worked best in my tests
            w_probs - path to the json file with word probabilites
                    - if not provided, standard average of word embeddings will be used
            compressed - indicates if a compressed FT model is being used - affects used model loading and word vector retrieving functions
    -   Added the update_embeddings call of the document_store inside init method, to allow successful retrieval
    Example of usage in model_test.py
    """

    def __init__(
            self,
            document_store: BaseDocumentStore,
            model_path: str,
            scale_score: bool = False,
            top_k: int = 10,
            progress_bar: bool = True,
            batch_size: int = 32,
            alpha: float = 1e-4,
            w_probs: str = None,
            compressed = False
    ):

        super().__init__()
        self.top_k = top_k
        self.document_store = document_store
        self.scale_score = scale_score
        self.progress_bar = progress_bar
        self.batch_size = batch_size

        if compressed:
            import compress_fasttext
            self.model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(model_path)
            self.get_w_vec = self.model.word_vec
        else:
            self.model = fasttext.load_model(model_path)
            self.get_w_vec = self.model.get_word_vector

        self.alpha = alpha
        self.word_probs = None
        if w_probs is not None:
            with open(w_probs, "r") as wp_file:
                self.word_probs = json.load(wp_file)
            self.embedding_func = self.weighted_embedding
        else:
            self.embedding_func = self.mean_embedding
        self.document_store.update_embeddings(retriever=self)

    def retrieve(self, query: str, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
                 top_k: Optional[int] = None, index: str = None, headers: Optional[Dict[str, str]] = None,
                 scale_score: bool = None) -> List[Document]:
        if top_k is None:
            top_k = self.top_k
        if index is None:
            index = self.document_store.index
        if scale_score is None:
            scale_score = self.scale_score
        query_emb = self.embed_queries(queries=[query])
        documents = self.document_store.query_by_embedding(
            query_emb=query_emb[0], filters=filters, top_k=top_k, index=index, headers=headers, scale_score=False
        )
        return documents

    def retrieve_batch(self, queries: List[str],
                       filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
                       top_k: Optional[int] = None, index: str = None, headers: Optional[Dict[str, str]] = None,
                       batch_size: Optional[int] = None, scale_score: bool = None) -> List[List[Document]]:
        if top_k is None:
            top_k = self.top_k

        if batch_size is None:
            batch_size = self.batch_size

        if isinstance(filters, list):
            if len(filters) != len(queries):
                raise HaystackError(
                    "Number of filters does not match number of queries. Please provide as many filters"
                    " as queries or a single filter that will be applied to each query."
                )
        else:
            filters = [filters] * len(queries) if filters is not None else [{}] * len(queries)

        if index is None:
            index = self.document_store.index
        if scale_score is None:
            scale_score = self.scale_score
        if not self.document_store:
            logger.error(
                "Cannot perform retrieve_batch() since EmbeddingRetriever initialized with document_store=None"
            )
            return [[] * len(queries)]  # type: ignore

        documents = []
        query_embs: List[np.ndarray] = []
        for batch in self._get_batches(queries=queries, batch_size=batch_size):
            query_embs.extend(self.embed_queries(queries=batch))
        for query_emb, cur_filters in tqdm(
                zip(query_embs, filters), total=len(query_embs), disable=not self.progress_bar, desc="Querying"
        ):
            cur_docs = self.document_store.query_by_embedding(
                query_emb=query_emb,
                top_k=top_k,
                filters=cur_filters,
                index=index,
                headers=headers,
                scale_score=False,
            )
            documents.append(cur_docs)
        return documents

    def mean_embedding(self, tokens: List[str]) -> np.ndarray:
        q_vectors = []
        for t in tokens:
            vector = self.get_w_vec(t) 
            q_vectors.append(vector/np.linalg.norm(vector))
        se = np.mean(q_vectors, axis=0)

        return se/np.linalg.norm(se)
    
    def weighted_embedding(self, tokens: List[str]) -> np.ndarray:
        def word_probability(word):
            if word in self.word_probs.keys():
                return self.word_probs[word]
            return 0.0

        wes = np.array([self.get_w_vec(t) for t in tokens])
        probs = np.array([word_probability(t) for t in tokens])[:, np.newaxis]
        wes /= np.linalg.norm(wes, axis=1)[:, np.newaxis] + 1e-9
        wes *= self.alpha / (self.alpha + probs)
        se = np.mean(wes, axis=0)
        return se/np.linalg.norm(se)

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        ret = []
        for query in queries:
            #tokens = word_tokenize(query)
            #tokens = [token.lower() for token in tokens if token.isalpha()] #if token.isalpha() left out because bibrams have _ in them
            tokens = query.lower().replace('\n', ' ').split()

            query_embedding = self.embedding_func(tokens)

            ret.append(query_embedding)

        return np.array(ret)

        # return np.array([self.model.get_sentence_vector(q.replace("\n", "")) for q in queries])

    def embed_documents(self, documents: List[Document]) -> np.ndarray:
        ret = []
        for doc in documents:
            #tokens = word_tokenize(doc.content)
            #tokens = [token.lower() for token in tokens if token.isalpha()]
            tokens = doc.content.lower().replace('\n', ' ').split()

            doc_embedding = self.embedding_func(tokens)

            ret.append(doc_embedding)

        return np.array(ret)