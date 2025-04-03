import os
import re
import math
from collections import defaultdict, Counter

class TextPreprocessor:
    def __init__(self):
        pass
    
    def tokenize(self, text):
        # Lowercase and split on alphabetic sequences only
        tokens = re.findall(r"[a-zA-Z]+", text.lower())
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


# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    preprocessor = TextPreprocessor()  # currently not using stopwords

    ################
    folder_path = "C:/Users/ppco9/OneDrive/바탕 화면/3-1 NLP & IR/textual_dataset"  # modify to the folder path where textual data are located
    ################

    documents = {}
    for i in range(1, 56):  # 1 through 35
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
    bool_searcher = BooleanSearcher(index)

    ##############
    # query_tokens_example = ["spacex", "AND", "landing", "OR", "spacex","AND", "starlink"]  # modify here for other queries
    query_tokens_example1 = ["space","AND","spacex"]  # modify here for other queries
    ##############
    
    # bool_result = bool_searcher.execute_query(query_tokens_example)
    bool_result1 = bool_searcher.execute_query(query_tokens_example1)
    # print(f"Boolean result for query {query_tokens_example}:", bool_result)  # shows the set containing the document ids
    print()
    print(f"Boolean result for query {query_tokens_example1}:", bool_result1)  # shows the set containing the document ids
    
    #--------------------------------------------
    # Vector Space Search Example
    # vec_searcher = VectorSpaceSearcher(index)

    # #############
    # query = "space rocket"  # modify here for other queries
    # #############

    # ranked_results = vec_searcher.search(query)
    # print(f"\nVector Space ranking for query '{query}':")
    # for doc_id, score in ranked_results[:10]:  # shows top 10 related results
    #     print(f"Doc {doc_id}: {score:.4f}")
