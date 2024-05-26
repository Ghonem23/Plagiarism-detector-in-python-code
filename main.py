import os
from numpy import vectorize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tempilary_files = [doc for doc in os.listdir() if doc.endswith('.txt')]
sample_contents = [open(text_input_file).read() for text_input_file in tempilary_files]


vectorize = lambda Text: TfidfVectorizer().fit_transform(Text).toarray()
similarity = lambda doc1, doc2: cosine_similarity([doc1, doc2])

vectors = vectorize(sample_contents)

s_vectors = list(zip(tempilary_files, vectors))

def check_plagiarism():
    results = set()
    global  s_vectors
    for temp_a, text_vector_a in s_vectors:
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((temp_a, text_vector_a))
        del new_vectors[current_index]
        for temp_b, text_vector_b in new_vectors:
            sim_score = similarity(text_vector_a, text_vector_b)[0][1]
            sample_pair = sorted((temp_a, temp_b))
            get_score = sample_pair[0], sample_pair[1], sim_score
            results.add(get_score)
    return results

for res_data in check_plagiarism():
    print(res_data)
