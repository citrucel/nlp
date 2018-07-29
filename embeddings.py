import numpy as np
from util import array_to_string
from util import text_prepare
from util import matrix_to_string
from sklearn.metrics.pairwise import cosine_similarity

DATA= './data/embeddings/'
embed_file= DATA + 'starspace_embedding.tsv'

def question_to_vec(question, embeddings, dim=300):
    """
        question: a string
        embeddings: dict where the key is a word and a value is its' embedding
        dim: size of the representation

        result: vector representation for the question
    """
    if question == '':
           return np.zeros(dim)
    scores = []
    for word in question.split():
        if word in embeddings:
            scores.append(embeddings[word])
    if not scores:
            return np.zeros(dim)
    return np.mean(scores, axis=0)

def hits_count(dup_ranks, k):
    """
        dup_ranks: list of duplicates' ranks; one rank per question;
                   length is a number of questions which we are looking for duplicates;
                   rank is a number from 1 to len(candidates of the question);
                   e.g. [2, 3] means that the first duplicate has the rank 2, the second one — 3.
        k: number of top-ranked elements (k in Hits@k metric)

        result: return Hits@k value for current ranking
    """
    a = np.array(dup_ranks)
    x = np.where(a <= k)
    return a[x].size/len(dup_ranks)


def dcgf(x):
    return 1 / np.log2(1 + x)


dcgf = np.vectorize(dcgf, otypes=[np.float])


def dcg_score(dup_ranks, k):
    """
        dup_ranks: list of duplicates' ranks; one rank per question;
                   length is a number of questions which we are looking for duplicates;
                   rank is a number from 1 to len(candidates of the question);
                   e.g. [2, 3] means that the first duplicate has the rank 2, the second one — 3.
        k: number of top-ranked elements (k in DCG@k metric)

        result: return DCG@k value for current ranking
    """
    a = np.array(dup_ranks)
    x = np.where(a <= k)
    return np.sum(dcgf(a[x])) / len(dup_ranks)



def read_corpus(filename):
    data = []
    for line in open(filename, encoding='utf-8'):
        data.append(line.strip().split('\t'))
    return data




def rank_candidates(question, candidates, embeddings, dim=300):
    """
        question: a string
        candidates: a list of strings (candidates) which we want to rank
        embeddings: some embeddings
        dim: dimension of the current embeddings

        result: a list of pairs (initial position in the list, question)
    """
    question_v = question_to_vec(question, embeddings, dim)
    question_v = question_v.reshape([1, -1])
    # print(question_v.shape)\
    candidates_v = np.array([question_to_vec(candidate, embeddings, dim) for candidate in candidates])
    candidates_v = candidates_v.reshape([len(candidates), -1])
    similarities = cosine_similarity(question_v, candidates_v)
    # print(similarities)
    ind = np.argsort(similarities)
    # print(ind)
    result = [(x, candidates[x]) for x in ind[0]][::-1]
    # print(result)
    return result

if __name__ == "__main__":
    validation = read_corpus(DATA + 'validation.tsv')
    prepared_validation = []
    for line in validation:
        new_line = [text_prepare(l) for l in line]
        prepared_validation.append(new_line)

    starspace_embeddings = {}
    for line in open(embed_file):
        a = line.strip().split('\\t')
        starspace_embeddings[a[0]] = [float(a[x]) for x in range(1, len(a))]

    ss_prepared_ranking = []
    for line in prepared_validation:
        q, *ex = line
        ranks = rank_candidates(q, ex, starspace_embeddings, 100)
#        ss_prepared_ranking.append([r[0] for r in ranks].index(0) + 1)
        ranked_candidates = [r[0] for r in ranks]
        ss_prepared_ranking.append([ranked_candidates.index(i) + 1 for i in range(len(ranked_candidates))])

    for k in [1, 5, 10, 100, 500, 1000]:
        print("DCG@%4d: %.3f | Hits@%4d: %.3f" % (k, dcg_score(ss_prepared_ranking, k),
                                                  k, hits_count(ss_prepared_ranking, k)))

    starspace_ranks_results = []
    prepared_test_data = DATA + 'prepared_test.tsv'
    for line in open(prepared_test_data):
        q, *ex = line.strip().split('\t')
        ranks = rank_candidates(q, ex, starspace_embeddings, 100)
        ranked_candidates = [r[0] for r in ranks]
        starspace_ranks_results.append([ranked_candidates.index(i) + 1 for i in range(len(ranked_candidates))])
