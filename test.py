import numpy as np
import csv

# y = np.asarray(['Java', 'C++', 'Other language', 'Python', 'C++', 'Python'])
# print(y)
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y_numeric = le.fit_transform(y)
# print(y_numeric)
# from sklearn.preprocessing import MultiLabelBinarizer
# mlb = MultiLabelBinarizer()
# y_indicator = mlb.fit_transform(y[:, None])
# print(y_indicator)
# y2 = np.asarray([['Java', 'Computer Vision'],
#                  ['C++', 'Speech Recognition'],
#                  ['Other language', 'Computer Vision'],
#                  ['Python', 'Other Application'],
#                  ['C++', 'Speech Recognition'],
#                  ['Python', 'Computer Vision']])

# def f(x):
#     return 1+ x
#     #return 1 / np.log2(1 + x)
# f = np.vectorize(f, otypes=[np.float])
#
# a = np.arange(1,10)
# print(a)
# x = np.where(a <= 3)
# print(f(a[x]))
# print(np.sum(f(a[x])))
#a = np.arange(4,10).reshape(2,3)
#x  = np.where(a>7)
#print(x)
#print(a[x])
def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """

    # Hint: you have already implemented a similar routine in the 3rd assignment.
    # Note that here you also need to know the dimension of the loaded embeddings.
    # When you load the embeddings, use numpy.float32 type as dtype

    ########################
    #### YOUR CODE HERE ####
    ########################
    embeddings = {}
    with open(embeddings_path, newline='') as embedding_file:
        # reader = csv.reader(embedding_file, delimiter='t')
        for line in embedding_file:
            nl = line.strip().split('\\t')
            word = nl[0]
            embedding = np.array(nl[1:]).astype(np.float32)
            embeddings[word] = embedding
        dim = len(line) - 1
        print(dim)

    print(len(embeddings))
    return embeddings

def maximum_gap( A):
    """find maximum gap between index(j -i ) with  A[i] <= A[j]"""
    gap = 0
    A = list(map( list,enumerate(A))) # get list of [index,value]
    # print(type(A))
    # for l in A:
    #     print(l)
    for item in A:
        item[0],item[1] = item[1], item[0] # swap index with value
    a = sorted(A)  # sort list A as per values

    max_index = a[0][1] # initialise max_index to first index in sorted list
    min_index = a[0][1] # initialise min_index to first index in sorted list

    for v,i in a:
        print(v,i)
        if i > max_index:  # if current > max_index, set max to current
            max_index = i
        if i < min_index:  # if current < min_index, set min to current
            min_index = i
            max_index = i  # reset max to current

        gap_new = max_index - min_index  # find the maximum gap
        if gap < gap_new:
            gap = gap_new

    return gap

#print(maximum_gap([3,5,4,2]))
#print(maximum_gap([-1,-1,2]))

DATA= 'natural-language-processing/project/data/'
load_embeddings(DATA + 'word_embeddings.tsv')