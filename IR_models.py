import os
import re
import math
import numpy as np  
import time
from collections import defaultdict, Counter
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

class TextPreprocessor:
    def __init__(self, remove_stopwords=False, stopwords=None):
        self.remove_stopwords = remove_stopwords
        self.stopwords = stopwords

    def tokenize(self, text):
        # Lowercase and split on alphabetic sequences only
        tokens = re.findall(r"[a-zA-Z]+|\d+|[^\w\s]", text.lower())
        if self.remove_stopwords and self.stopwords:   
            tokens = [token for token in tokens if token not in self.stopwords]
        return tokens
    
class InvertedIndex:
    def __init__(self, preprocessor):
        self.postings = defaultdict(set)
        self.term_frequencies = defaultdict(lambda: defaultdict(int))
        self.doc_frequency = defaultdict(int)
        self.num_docs = 0
        self.preprocessor = preprocessor
    
    def build_index(self, documents):
        self.num_docs = len(documents)
        
        for doc_id, text in documents.items():
            tokens = self.preprocessor.tokenize(text)
            unique_terms = set()
            
            for token in tokens:
                self.postings[token].add(doc_id)
                self.term_frequencies[token][doc_id] += 1
                unique_terms.add(token)
            
            for term in unique_terms:
                self.doc_frequency[term] += 1

    def get_postings(self, term):
        return self.postings.get(term, set())

class BooleanSearcher:
    def __init__(self, inverted_index):
        self.index = inverted_index
    
    def boolean_and(self, term1, term2):
        return self.index.get_postings(term1) & self.index.get_postings(term2)
    
    def boolean_or(self, term1, term2):
        return self.index.get_postings(term1) | self.index.get_postings(term2)
    
    def boolean_not(self, term):
        all_docs = set(range(1, self.index.num_docs + 1))
        return all_docs - self.index.get_postings(term)
    
    def execute_query(self, query_tokens):
        if not query_tokens:
            return set()
        
        def get_postings_for_token(t):
            return self.index.get_postings(t)
        
        idx = 0
        current_set = set()
        
        # If first token is "NOT" : process the first NOT separately
        if query_tokens[0].lower() == "not":
            not_postings = self.boolean_not(query_tokens[1])
            current_set = not_postings
            idx = 2
        else:
            current_set = get_postings_for_token(query_tokens[0])
            idx = 1

        while idx < len(query_tokens):
            operator = query_tokens[idx].lower()
            idx += 1
            
            if idx < len(query_tokens) and query_tokens[idx].lower() == "not":
                idx += 1
                term = query_tokens[idx]
                new_set = self.boolean_not(term)
            else:
                term = query_tokens[idx]
                new_set = get_postings_for_token(term)
            
            idx += 1
            
            if operator == "and":
                current_set = current_set & new_set
            elif operator == "or":
                current_set = current_set | new_set
            else:
                # default to AND op
                current_set = current_set & new_set
        
        return current_set

class VectorSpaceSearcher:
    def __init__(self, inverted_index):
        self.index = inverted_index

    def compute_query_vector(self, query):
        tokens = self.index.preprocessor.tokenize(query)
        token_counts = Counter(tokens)
        query_vector = {}
        N = self.index.num_docs
        
        for term, count in token_counts.items():
            df_t = self.index.doc_frequency.get(term, 0)
            if df_t == 0:
                continue
            tf  = 1 + math.log(count, 10)
            idf = math.log((N / df_t), 10)
            query_vector[term] = tf * idf
        return query_vector
    
    def compute_doc_vector_norm(self, doc_id):
        total = 0.0
        N = self.index.num_docs
        for term, freqs in self.index.term_frequencies.items():
            if doc_id in freqs:
                tf_doc = freqs[doc_id]
                tf_weight = 1 + math.log(tf_doc, 10)
                df_t = self.index.doc_frequency[term]
                idf = math.log((N / df_t), 10)
                tf_idf = tf_weight * idf
                total += (tf_idf ** 2)
        return math.sqrt(total)

    def compute_cosine_similarity(self, query_vector, doc_id):
        dot = 0.0
        N = self.index.num_docs
        norm_query = math.sqrt(sum(val * val for val in query_vector.values()))
        norm_doc = self.compute_doc_vector_norm(doc_id)
        
        if norm_query == 0 or norm_doc == 0:
            return 0.0
        
        for term, q_val in query_vector.items():
            if doc_id in self.index.term_frequencies[term]:
                tf_doc = self.index.term_frequencies[term][doc_id]
                tf_weight = 1 + math.log(tf_doc, 10)
                df_t = self.index.doc_frequency[term]
                idf = math.log((N / df_t), 10)
                doc_val = tf_weight * idf
                dot += q_val * doc_val
        
        return dot / (norm_query * norm_doc)

    def search(self, query):
        query_vector = self.compute_query_vector(query)
        scores = []
        all_doc_ids = range(1, self.index.num_docs + 1)
        for doc_id in all_doc_ids:
            score = self.compute_cosine_similarity(query_vector, doc_id)
            scores.append((doc_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

class LSISearcher:
    """
    Latent‐Semantic Indexing on top of the existing InvertedIndex.

    Build once → query many times.
    """
    def __init__(self, inverted_index, k=100, use_tfidf=True):
        self.index          = inverted_index          # already built
        self.k              = k                       # latent dims
        self.use_tfidf      = use_tfidf
        self._build_matrices()
        self._fit_svd()

    # ------------------------------------------------------------------
    # --------------  PHASE 1  :  BUILD ORIGINAL SPACE  -----------------
    # ------------------------------------------------------------------
    def _build_matrices(self):
        """
        Create a |V| × |D| matrix *A* where rows = vocabulary terms
        and cols = documents.  A[i,j] is tf or tf–idf weight.
        """
        terms        = list(self.index.postings.keys())
        term_to_row  = {t:i for i,t in enumerate(terms)}
        docs         = range(1, self.index.num_docs + 1)

        self.A = np.zeros((len(terms), self.index.num_docs), dtype=float)

        N = self.index.num_docs
        for term,row in term_to_row.items():
            df = self.index.doc_frequency[term]
            idf = math.log((N / (df + 1)), 10)  # N / df + 1 
            for doc_id, tf in self.index.term_frequencies[term].items():
                w = 1 + math.log(tf, 10)
                self.A[row, doc_id-1] = w * idf if self.use_tfidf else w

        # store for later lookup
        self._terms = terms
        self._term_to_row = term_to_row
        # normalise document columns (as in classic VSM)
        self.A = normalize(self.A, axis=0)

    # ------------------------------------------------------------------
    # --------------  PHASE 2  :  SVD  &  REDUCTION  --------------------
    # ------------------------------------------------------------------
    def _fit_svd(self):
        """
        Perform truncated SVD:  A ≈ U_k Σ_k V_kᵀ
        Keep k singular vectors → reduced coordinates.
        """
        svd = TruncatedSVD(n_components=self.k, n_iter=7, random_state=42)
        self.U_k  = svd.fit_transform(self.A)          # |V| × k
        self.S_k  = svd.singular_values_              # length-k diag (1-D)
        self.V_kT = svd.components_                   # k × |D|
        # σ⁻¹  for later projection
        self.S_inv = np.diag(1.0 / self.S_k)

        # Pre-compute document vectors in latent space:  D_k = Σ_k V_kᵀᵀ
        self.docs_latent = (self.S_k[:, None] * self.V_kT).T   # |D| × k
        self.docs_latent = normalize(self.docs_latent, axis=1)

    # ------------------------------------------------------------------
    # --------------------  PHASE 3  :  QUERY  --------------------------
    # ------------------------------------------------------------------
    def _transform_query(self, query):
        """
        Map raw query → latent space:
            q   →  tf-idf column vector (|V|×1)
            q_k = Σ⁻¹ Uᵀ q
        """
        tokens = self.index.preprocessor.tokenize(query)
        if not tokens:
            return None

        q_vec = np.zeros((len(self._terms),), dtype=float)
        counts = Counter(tokens)
        N = self.index.num_docs
        for term, c in counts.items():
            row = self._term_to_row.get(term)
            if row is None:
                continue
            tf = 1 + math.log(c, 10)
            df = self.index.doc_frequency[term]
            idf = math.log((N / df), 10) if df else 0.0
            q_vec[row] = tf * idf if self.use_tfidf else tf
        if np.all(q_vec == 0):
            return None

        q_vec = normalize(q_vec.reshape(1, -1))[0]

        # latent projection: q_k = (Σ⁻¹ Uᵀ) q
        q_latent = self.S_inv @ (self.U_k.T @ q_vec)
        return normalize(q_latent.reshape(1, -1))[0]

    # ------------------------------------------------------------------
    # --------------------  PUBLIC API  ---------------------------------
    # ------------------------------------------------------------------
    def search(self, query, top_k=10):
        """
        Return a list of (doc_id, score) sorted by cosine similarity
        in latent space.
        """
        q_latent = self._transform_query(query)
        if q_latent is None:
            return []

        # cosine = dot product because both sides are length-1
        sims = self.docs_latent @ q_latent  # = projecting q_k onto D_k : captures "conceptual" closeness
        # sims is |D|-vector in doc-id order 1…N
        ranked = sorted(enumerate(sims, start=1),
                        key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    stopwords = {
    "the", "and", "or", "of", "a", "to", "in", "for", "on", "with", "at",
    "by", "an", "be", "is", "are", "was", "were", "from",
    "this", "that", "it", "as", "but", "not",
    "has", "have", "had",
    "will", "would", "can", "could", "shall", "should", "may", "might", "must",
    "i", "you", "he", "she", "they", "we",
    "my", "your", "his", "her", "its", "their", "our",
    "which", "who", "whom", "whose", "what", "where", "when", "why", "how",
    "if", "then", "else",
    "about", "into", "through", "during", "before", "after", "above", "below"
}
    
    preprocessor = TextPreprocessor(remove_stopwords=True, stopwords=stopwords)  # currently not using stopwords

    ################
    folder_path = "C:/Users/ppco9/OneDrive/바탕 화면/3-1 NLP & IR/textual_dataset"  # modify to the folder path where textual data are located
    ################

    documents = {}
    for i in range(1, 56):  # 1 to 55
        file_path = os.path.join(folder_path, f"{i}.txt")
        with open(file_path, "r", encoding="utf-8") as f:
            documents[i] = f.read()

    index = InvertedIndex(preprocessor)
    index.build_index(documents)



    # Words I suggest you can try :
    '''
    space, planet, moon, rocket, glacier, f1, coronavirus, ice, star, NASA, AI, scientists
    '''

    # Boolean Search example
    # bool_searcher = BooleanSearcher(index)

    # ##############
    # # query_tokens_example = ["spacex", "AND", "landing", "OR", "spacex","AND", "starlink"]  # modify here for other queries
    # query_tokens_example1 = ["space","AND","spacex"]  # modify here for other queries
    # ##############
    
    # # bool_result = bool_searcher.execute_query(query_tokens_example)
    # bool_result1 = bool_searcher.execute_query(query_tokens_example1)
    # # print(f"Boolean result for query {query_tokens_example}:", bool_result)  # shows the set containing the document ids
    # print()
    # print(f"Boolean result for query {query_tokens_example1}:", bool_result1)  # shows the set containing the document ids
    
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------
    

    ## Vector Space Search vs LSI Search
    #----------------------setup-----------------------
    start = time.time()
    vec_searcher = VectorSpaceSearcher(index)
    end = time.time()
    print(f"Time taken for Vector Space model setup: {end - start:.4f} seconds")

    start = time.time()
    lsi_searcher = LSISearcher(index, k=20)  
    end = time.time()
    print(f"Time taken for LSI model setup: {end - start:.4f} seconds\n")

    #---------------------search speed-----------------------
    query = "spaceX conducting new engine tests"  # modify here for other queries
    
    start = time.time()
    vsm_ranked_results = vec_searcher.search(query)
    end = time.time()
    print(f"Time taken for Vector Space Search: {end - start:.4f} seconds")

    start = time.time()
    lsi_ranked_results = lsi_searcher.search(query)
    end = time.time()
    print(f"Time taken for LSI Search: {end - start:.4f} seconds\n")

    #---------------------ranking-----------------------
    print(f"Vector Space ranking for query '{query}':")
    for doc_id, score in vsm_ranked_results[:10]:  # shows top 10 related results
        print(f"Doc {doc_id}: {score:.4f}")

    print(f"\nLSI ranking for query '{query}':")
    for doc_id, score in lsi_ranked_results[:10]:
        print(f"Doc {doc_id}: {score:.4f}")
