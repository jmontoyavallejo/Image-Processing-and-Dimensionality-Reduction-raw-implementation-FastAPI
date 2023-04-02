import pandas as pd
import numpy as np

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.eigenvectors = None

    def fit(self, X):
        mean_centered = X - X.mean(axis=0)
        cov_matrix = np.cov(mean_centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        eigenvectors = eigenvectors.T
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvectors = eigenvectors[sorted_indices[:self.n_components]]

    def fit_transform(self, X):
        self.fit(X)
        mean_centered = X - X.mean(axis=0)
        return np.dot(mean_centered, self.eigenvectors.T)

    def transform(self, X):
        mean_centered = X - X.mean(axis=0)
        return np.dot(mean_centered, self.eigenvectors.T)
    
    
class SVC:
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
