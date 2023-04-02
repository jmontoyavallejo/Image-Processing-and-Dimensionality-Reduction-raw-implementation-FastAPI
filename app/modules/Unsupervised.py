import pandas as pd
import numpy as np
from keras.datasets import mnist
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt



class BaseEstimator:
    y_required = True
    fit_required = True

    def _setup_input(self, X, y=None):
        """Ensure inputs to an estimator are in the expected format.
        Ensures X and y are stored as numpy ndarrays by converting from an
        array-like object if necessary. Enables estimators to define whether
        they require a set of y target values or not with y_required, e.g.
        kmeans clustering requires no target labels and is fit against only X.
        Parameters
        ----------
        X : array-like
            Feature dataset.
        y : array-like
            Target values. By default is required, but if y_required = false
            then may be omitted.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.size == 0:
            raise ValueError("Got an empty matrix.")

        if X.ndim == 1:
            self.n_samples, self.n_features = 1, X.shape
        else:
            self.n_samples, self.n_features = X.shape[0], np.prod(X.shape[1:])

        self.X = X

        if self.y_required:
            if y is None:
                raise ValueError("Missed required argument y")

            if not isinstance(y, np.ndarray):
                y = np.array(y)

            if y.size == 0:
                raise ValueError("The targets array must be no-empty.")

        self.y = y

    def fit(self, X, y=None):
        self._setup_input(X, y)

    def predict(self, X=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if self.X is not None or not self.fit_required:
            return self._predict(X)
        else:
            raise ValueError("You must call `fit` before `predict`")

    def _predict(self, X=None):
        raise NotImplementedError()

class PCA(BaseEstimator):
    y_required = False

    def __init__(self, n_components, solver="svd"):
        """Principal component analysis (PCA) implementation.
        Transforms a dataset of possibly correlated values into n linearly
        uncorrelated components. The components are ordered such that the first
        has the largest possible variance and each following component as the
        largest possible variance given the previous components. This causes
        the early components to contain most of the variability in the dataset.
        Parameters
        ----------
        n_components : int
        solver : str, default 'svd'
            {'svd', 'eigen'}
        """
        self.solver = solver
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X, y=None):
        self.mean = np.mean(X, axis=0)
        self._decompose(X)

    def _decompose(self, X):
        # Mean centering
        X = X.copy()
        X -= self.mean

        if self.solver == "svd":
            _, s, Vh = SVD(X, full_matrices=True)
        elif self.solver == "eigen":
            s, Vh = np.linalg.eig(np.cov(X.T))
            Vh = Vh.T

        s_squared = s ** 2
        variance_ratio = s_squared / s_squared.sum()
        self.components = Vh[0: self.n_components]

    def transform(self, X):
        X = X.copy()
        X -= self.mean
        return np.dot(X, self.components.T)

    def _predict(self, X=None):
        return self.transform(X)
    
    
class SVD:
    def __init__(self, C=1.0, kernel='rbf', gamma='scale'):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        alpha = np.zeros(n_samples)
        gram_matrix = self._compute_gram_matrix(X)
        for _ in range(1000):
            for i in range(n_samples):
                error_i = np.dot(self.weights, gram_matrix[i]) + self.bias - y[i]
                alpha_i_old = alpha[i]
                alpha_i = alpha_i_old + error_i / gram_matrix[i,i]
                alpha_i = max(0, min(alpha_i, self.C))
                alpha[i] = alpha_i
                self.weights += (alpha_i - alpha_i_old) * y[i] * X[i]
                self.bias += y[i] - np.dot(self.weights, X[i])
    
    def _compute_gram_matrix(self, X):
        return np.dot(X, X.T)

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.bias)
    

class TSNE:
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate

    def fit_transform(self, X):
        n_samples = X.shape[0]
        distances = self._compute_pairwise_distances(X)
        P = self._compute_conditional_probabilities(distances)
        Y = np.random.normal(size=(n_samples, self.n_components))
        for _ in range(1000):
            grad = self._compute_gradient(P, Y)
            Y -= self.learning_rate * grad
        return Y

    def _compute_pairwise_distances(self, X):
        sum_X = np.sum(np.square(X), axis=1)
        distances = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        return distances
    def _compute_conditional_probabilities(self, distances):
        P = np.zeros_like(distances)
        for i in range(distances.shape[0]):
            sorted_indices = np.argsort(distances[i])
            distances_i = distances[i, sorted_indices[1:self.perplexity+1]]
            numerator = np.exp(-distances_i ** 2 / (2 * self._get_variance(distances_i)))
            P[i, sorted_indices[1:self.perplexity+1]] = numerator / np.sum(numerator)
        return (P + P.T) / (2 * distances.shape[0])

    def _compute_gradient(self, P, Y):
        Q = self._compute_similarity_matrix(Y)
        PQ_diff = P - Q
        PQ_diff_diag = np.diag(PQ_diff)
        t_grad = np.dot(PQ_diff, Y) - np.dot(Y, np.diag(PQ_diff_diag))
        return 4 * t_grad * (1 + np.square(np.linalg.norm(Y, axis=1, keepdims=True))) ** -1

    def _compute_similarity_matrix(self, Y):
        distances = self._compute_pairwise_distances(Y)
        inv_distances = (1 + distances) ** -1
        np.fill_diagonal(inv_distances, 0)
        return inv_distances / np.sum(inv_distances)

    def _get_variance(self, distances):
        return np.square(distances).sum() / distances.size

    def transform(self, X):
        return self.fit_transform(X)


def Train_model_scikit_learn(x_train,y_train,x_test,y_test):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

def load_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    train_mask = np.isin(y_train, [0, 8])
    test_mask = np.isin(y_test, [0, 8])
    x_train = x_train[train_mask]
    y_train = y_train[train_mask]
    x_test = x_test[test_mask]
    y_test = y_test[test_mask]

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    return x_train,y_train,x_test,y_test


def plot_PCA_TSNE_unsupervised_module():
    x_train,y_train,_,_=load_mnist_dataset()
    color=['r' if i==0 else 'b' for i in y_train ]
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(x_train)
    tsne = TSNE(n_components=2)
    tsne_components = tsne.fit_transform(x_train)
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.scatter(pca_components[:, 0], pca_components[:, 1], c=color)
    ax1.set_title('PCA')
    ax2.scatter(tsne_components[:, 0], tsne_components[:, 1], c=color)
    ax2.set_title('t-SNE')
    plt.savefig('app/resources/pca_vs_tsne_my_unsupervised_module.png')

def PCA_Training_scikit_unsupervised_module(x_train,y_train,x_test,y_test):
    pca = PCA(n_components=2)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    model = LogisticRegression()
    model.fit(x_train_pca, y_train)
    return model.score(x_test_pca, y_test)


def TSNE_Training_unsupervised_module(x_train,y_train,x_test,y_test):
    tsne = TSNE(n_components=2)
    x_train_tsne = tsne.fit_transform(x_train)
    x_test_tsne = tsne.fit_transform(x_test)
    model = LogisticRegression()
    model.fit(x_train_tsne, y_train)
    return model.score(x_test_tsne, y_test)