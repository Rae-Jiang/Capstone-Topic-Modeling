from sklearn.preprocessing import scale
from sklearn.linear_model import Lasso
from tqdm import tqdm_notebook as tqdm
from matplotlib import pyplot as plt
import networkx as nx
from collections import Counter

def prepare_CTM_data(document_term_matrix):
    N, V = document_term_matrix.shape
    list_output_row = [f'{row.nnz} '+' '.join([f'{word_idx}:{count_idx}' for (_, word_idx), count_idx in row.todok().items()]) for row in document_term_matrix]
    return list_output_row

def print_topics(beta_file, vocab_file, nwords = 25,):
    vocab = open(vocab_file, 'r').read().split('\n')
    indices = np.array(list(range(len(vocab))))
    topic = np.array(list(map(float, open(beta_file, 'r').readlines())))
    nterms  = len(vocab)
    ntopics = int(len(topic)/nterms)
    topic   = np.reshape(topic, [ntopics, nterms])
    output = []
    for i in range(ntopics):
        word_dist = topic[i]
        top_n_word_idx = word_dist.argsort()[::-1][:nwords]
        print(f'Topic {i}: ', ' '.join([vocab[i] for i in top_n_word_idx]))
        output.append('\n'.join([vocab[i] for i in top_n_word_idx]))
    return output

def build_lasso_graph(x, l1_lambda, topic_words,both=True,):
    """
    A adaption from CTM's lasso-graph.r script
    
    Parameters
    ----------
    x: numpy.array
        N x K data matrix -- e.g., the variational means ("final-lambda.dat")
    l1_lambda: float
        relative bound on the l1-norm of the parameters, in [0,1]
    and: bool
        if and=T/F then the graph is computed by taking the intersction/union of the nbhds
        
    Returns
    -------
    ihat: numpy.array 
        K x K adjacency matrix of the topic graph
    """
    x = scale(x)
    topic_count = Counter(x.argmax(axis=1))
    N, K = x.shape
    Shat = np.zeros((K,K), dtype=bool)
    print('Parameters:')
    print(f'N={N}, K={K}, lambda={l1_lambda}')
    print()
    print('Fitting...')
    for j in range(K):
        column_mask = np.ones(20, dtype=bool)
        column_mask[j] = False
        # The response is the j-th column
        y = x[:,j]
        X = x[:,column_mask]
        # Do the l1-regularized regression
        # Note: the bound in l1ce code is the upper bound on the l1
        # norm.  So, a larger bound is a weaker constraint on the model
        lasso_model = Lasso(
            normalize=False,
            alpha=l1_lambda,
            tol=1e-6,
            #positive=True,
            max_iter=10000,
        )
        lasso_model.fit(X, y)
        indices = np.array(range(K))[column_mask]
        beta = lasso_model.coef_
        nonzero = indices[beta>0]
        Shat[j, nonzero] = True
        Shat[j,j] = True
    print('Fitting completed!')
    # Include an edge if either (and=F) or both (and=T) endpoints are neighbors
    Ihat = np.zeros((K,K), dtype=bool)
    if both is True:
        for i in range(K):
            Ihat[:, i] = Shat[:,i]&Shat[i,:]
    else:
        for i in range(K):
            Ihat[:, i] = Shat[:,i]|Shat[i,:]
    # Visualize topic graph
    # Construct Graph from adjacency matrix
    G = nx.from_numpy_array(Ihat)
    annotations = [f'({topic_count[i]*100/N}%)\n'+topic_word for i, topic_word in enumerate(topic_words)]
    node_label_dict = dict(zip(range(20), annotations))
    pos = nx.spring_layout(G, k=0.4)
    plt.figure(figsize=(20,20))
    plt.title(f'Number of topics: {K}, L1-Regularization Strength: {l1_lambda}')
    nx.draw(G, pos, font_size=10, with_labels=True, labels=node_label_dict, node_size=5)
    return Ihat, fig