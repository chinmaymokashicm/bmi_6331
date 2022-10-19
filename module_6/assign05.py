"""
Author: Luca Giancardo
Date: 2017-10-11
Version: 1.0

part of the code adapted from:
http://sebastianraschka.com/Articles/2014_python_lda.html
"""


import numpy as np



def lda(X,y, n_components, debug=False):
    """
    Compute Linear Discriminant Analysis on X according to the classes y and reduce samples to n_components.
    :param X: feature matrix (samples x features)
    :param y: class vector (must be numerical)
    :param n_components: number of compoents to project to
    :param debug: set to true to activate printout
    :return: (lower dimensional samples, W: eigen vectors used for the projection)
    """


    # param
    featDim = X.shape[1]
    classVec = np.unique(y)


    # mean vectors
    mean_vectors = []
    for cl in classVec:
        mean_vectors.append(np.mean(X[y == cl], axis=0))
        if debug:
            print('Mean Vector class %s: %s\n' % (cl, mean_vectors[cl - 1]))

    # scatter matrices
    S_W = np.zeros((featDim, featDim))
    for cl, mv in zip(classVec, mean_vectors):
        class_sc_mat = np.zeros((featDim, featDim))  # scatter matrix for every class
        for row in X[y == cl]:
            row, mv = row.reshape(featDim, 1), mv.reshape(featDim, 1)  # make column vectors
            class_sc_mat += (row - mv).dot((row - mv).T)
        S_W += class_sc_mat  # sum class scatter matrices
    if debug:
        print('within-class Scatter Matrix:\n', S_W)

    # Between class scatter matrix
    overall_mean = np.mean(X, axis=0)

    S_B = np.zeros((featDim, featDim))
    for i, mean_vec in enumerate(mean_vectors):
        n = X[y == i + 1, :].shape[0]
        mean_vec = mean_vec.reshape(featDim, 1)  # make column vector
        overall_mean = overall_mean.reshape(featDim, 1)  # make column vector
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    if debug:
        print('between-class Scatter Matrix:\n', S_B)

    # generalized eigen problem
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

    for i in range(len(eig_vals)):
        eigvec_sc = eig_vecs[:, i].reshape(featDim, 1)
        if debug:
            print('\nEigenvector {}: \n{}'.format(i + 1, eigvec_sc.real))
            print('Eigenvalue {:}: {:.2e}'.format(i + 1, eig_vals[i].real))


    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

    # Visually confirm that the list is correctly sorted by decreasing eigenvalues
    if debug:
        print('Eigenvalues in decreasing order:\n')
        for i in eig_pairs:
            print(i[0])
    if debug:
        print('Variance explained:\n')
        eigv_sum = sum(eig_vals)
        for i, j in enumerate(eig_pairs):
            print('eigenvalue {0:}: {1:.2%}'.format(i + 1, (j[0] / eigv_sum).real))

    # select n components
    Wlst = []
    for i in range(n_components):
        Wlst.append( eig_pairs[i][1].reshape(featDim, 1) )
        # W = np.hstack(( eig_pairs[0][1].reshape(featDim, 1), eig_pairs[1][1].reshape(featDim, 1)))
    # create matrix
    W = np.array(Wlst)

    # project
    X_lda = X.dot(W)

    return X_lda, W

if __name__ == "__main__":
    pass

