import cv2
import numpy as np
from keras.datasets import mnist
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from numpy.linalg import eigh,norm
import logging
import pickle

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

class PCA_unsupervised_module(BaseEstimator):
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

    def fit_transform(self, X, y=None):
        self.fit(X) 
        return self.transform(X)

    def _decompose(self, X):
        # Mean centering
        X = X.copy()
        X -= self.mean

        if self.solver == "svd":
            _, s, Vh = SVD_unsupervised_module(X)
        elif self.solver == "eigen":
            s, Vh = np.linalg.eig(np.cov(X.T))
            Vh = Vh.T

        s_squared = s ** 2
        variance_ratio = s_squared / s_squared.sum()
        logging.info("Explained variance ratio: %s" % (variance_ratio[0: self.n_components]))
        self.components = Vh[0: self.n_components]

    def transform(self, X):
        X = X.copy()
        X -= self.mean
        return np.dot(X, self.components.T)

    def _predict(self, X=None):
        return self.transform(X)   


class TSNE_unsupervise_module():
  def __init__(self, n_components=2, perplexity=20.0, max_iter=500, learning_rate=10,random_state=1):
    """A t-Distributed Stochastic Neighbor Embedding implementation.
    Parameters
    ----------
    max_iter : int, default 200
    perplexity : float, default 30.0
    n_components : int, default 2
    """
    self.max_iter = max_iter
    self.perplexity = perplexity
    self.n_components = n_components
    self.momentum = 0.9
    self.lr = learning_rate
    self.seed=random_state

  def fit_transform(self,X):
    self.fit(X)
    return self.transform(X)

  def fit(self,X):
    self.Y = np.random.RandomState(self.seed).normal(0., 0.0001, [X.shape[0], self.n_components])
    self.Q, self.distances = self.q_tsne()
    self.P=self.p_joint(X)

  def transform(self,X):
    if self.momentum:
        Y_m2 = self.Y.copy()
        Y_m1 = self.Y.copy()

    for i in range(self.max_iter):

        # Get Q and distances (distances only used for t-SNE)
        self.Q, self.distances = self.q_tsne()
        # Estimate gradients with respect to Y
        grads = self.tsne_grad()

        # Update Y
        self.Y = self.Y - self.lr * grads

        if self.momentum:  # Add momentum
            self.Y += self.momentum * (Y_m1 - Y_m2)
            # Update previous Y's for momentum
            Y_m2 = Y_m1.copy()
            Y_m1 = self.Y.copy()
    return self.Y

  def p_joint(self,X):
    """Given a data matrix X, gives joint probabilities matrix.
    # Arguments
        X: Input data matrix.
    # Returns:
        P: Matrix with entries p_ij = joint probabilities.
    """
    def p_conditional_to_joint(P):
      """Given conditional probabilities matrix P, return
      approximation of joint distribution probabilities."""
      return (P + P.T) / (2. * P.shape[0])
    def calc_prob_matrix(distances, sigmas=None, zero_index=None):
      """Convert a distances matrix to a matrix of probabilities."""
      if sigmas is not None:
          two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))
          return self.softmax(distances / two_sig_sq, zero_index=zero_index)
      else:
          return self.softmax(distances, zero_index=zero_index)
    # Get the negative euclidian distances matrix for our data
    distances = self.neg_squared_euc_dists(X)
    # Find optimal sigma for each row of this distances matrix
    sigmas = self.find_optimal_sigmas()
    # Calculate the probabilities based on these optimal sigmas
    p_conditional = calc_prob_matrix(distances, sigmas)
    # Go from conditional to joint probabilities matrix
    self.P = p_conditional_to_joint(p_conditional)
    return self.P
  

  def find_optimal_sigmas(self):
    """For each row of distances matrix, find sigma that results
    in target perplexity for that role."""
    def binary_search(eval_fn, target, tol=1e-10, max_iter=10000,
                  lower=1e-20, upper=1000.):
      """Perform a binary search over input values to eval_fn.
      # Arguments
          eval_fn: Function that we are optimising over.
          target: Target value we want the function to output.
          tol: Float, once our guess is this close to target, stop.
          max_iter: Integer, maximum num. iterations to search for.
          lower: Float, lower bound of search range.
          upper: Float, upper bound of search range.
      # Returns:
          Float, best input value to function found during search.
      """
      for i in range(max_iter):
          guess = (lower + upper) / 2.
          val = eval_fn(guess)
          if val > target:
              upper = guess
          else:
              lower = guess
          if np.abs(val - target) <= tol:
              break
      return guess
    def calc_perplexity(prob_matrix):
      """Calculate the perplexity of each row
      of a matrix of probabilities."""
      entropy = -np.sum(prob_matrix * np.log2(prob_matrix), 1)
      perplexity = 2 ** entropy
      return perplexity

    def perplexity(distances, sigmas, zero_index):
        """Wrapper function for quick calculation of
        perplexity over a distance matrix."""
        def calc_prob_matrix(distances, sigmas=None, zero_index=None):
          """Convert a distances matrix to a matrix of probabilities."""
          if sigmas is not None:
              two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))
              return self.softmax(distances / two_sig_sq, zero_index=zero_index)
          else:
              return self.softmax(distances, zero_index=zero_index)
        return calc_perplexity(
            calc_prob_matrix(distances, sigmas, zero_index))
    sigmas = []
    # For each row of the matrix (each point in our dataset)
    for i in range(self.distances.shape[0]):
        # Make fn that returns perplexity of this row given sigma
        eval_fn = lambda sigma: \
            perplexity(self.distances[i:i+1, :], np.array(sigma), i)
        # Binary search over sigmas to achieve target perplexity
        correct_sigma = binary_search(eval_fn, self.perplexity)
        # Append the resulting sigma to our output array
        sigmas.append(correct_sigma)
    return np.array(sigmas)


  def tsne_grad(self):
    """t-SNE: Estimate the gradient of the cost with respect to Y."""
    pq_diff = self.P - self.Q  # NxN matrix
    pq_expanded = np.expand_dims(pq_diff, 2)  # NxNx1
    y_diffs = np.expand_dims(self.Y, 1) - np.expand_dims(self.Y, 0)  # NxNx2
    # Expand our distances matrix so can multiply by y_diffs
    distances_expanded = np.expand_dims(self.distances, 2)  # NxNx1
    # Weight this (NxNx2) by distances matrix (NxNx1)
    y_diffs_wt = y_diffs * distances_expanded  # NxNx2
    grad = 4. * (pq_expanded * y_diffs_wt).sum(1)  # Nx2
    return grad

  def neg_squared_euc_dists(self,X):
      """Compute matrix containing negative squared euclidean
      distance for all pairs of points in input matrix X
      # Arguments:
          X: matrix of size NxD
      # Returns:
          NxN matrix D, with entry D_ij = negative squared
          euclidean distance between rows X_i and X_j
      """
      # Math? See https://stackoverflow.com/questions/37009647
      sum_X = np.sum(np.square(X), 1)
      D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
      return -D


  def softmax(self,X, diag_zero=True, zero_index=None):
      """Compute softmax values for each row of matrix X."""

      # Subtract max for numerical stability
      e_x = np.exp(X - np.max(X, axis=1).reshape([-1, 1]))

      # We usually want diagonal probailities to be 0.
      if zero_index is None:
          if diag_zero:
              np.fill_diagonal(e_x, 0.)
      else:
          e_x[:, zero_index] = 0.

      # Add a tiny constant for stability of log we take later
      e_x = e_x + 1e-8  # numerical stability

      return e_x / e_x.sum(axis=1).reshape([-1, 1])

  def q_tsne(self):
    """t-SNE: Given low-dimensional representations Y, compute
    matrix of joint probabilities with entries q_ij."""
    distances = self.neg_squared_euc_dists(self.Y)
    inv_distances = np.power(1. - distances, -1)
    np.fill_diagonal(inv_distances, 0.)
    return inv_distances / np.sum(inv_distances), inv_distances
    

def SVD_unsupervised_module(A):
  ev,V=eigh(A.T@A)
  u=[]
  for i in range(A.shape[1]):
    u.append(A@V[:,i]/norm(A@V[:,i]))
  U=np.array(u).T
  S=np.round(U.T@A@V,decimals=5)
  return U,np.diagonal(S),V.T


def Train_model_scikit_learn(x_train,y_train,x_test,y_test):
    model = LogisticRegression(random_state=1111)
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)


def load_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    train_mask_0 = np.isin(y_train, [0])
    train_mask_8 = np.isin(y_train, [8])
    x_train_0 = x_train[train_mask_0]
    x_train_8 = x_train[train_mask_8]
    x_train=np.concatenate((x_train_0[:500,:,:],x_train_8[:500,:,:]))
    y_train=np.concatenate((y_train[train_mask_0][:500],y_train[train_mask_8][:500]))
    
    test_mask_0 = np.isin(y_test, [0])
    test_mask_8 = np.isin(y_test, [8])
    x_test_0 = x_test[test_mask_0]
    x_test_8 = x_test[test_mask_8]
    x_test=np.concatenate((x_test_0[:200,:,:],x_test_8[:200,:,:]))
    y_test=np.concatenate((y_test[test_mask_0][:200],y_test[test_mask_8][:200]))

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return x_train,y_train,x_test,y_test


def plot_PCA_TSNE_unsupervised_module():
    x_train,y_train,x_test,y_test=load_mnist_dataset()
    color_train=['r' if i==0 else 'b' for i in y_train ]
    color_test = ['r' if i == 0 else 'b' for i in y_test]
    pca = PCA_unsupervised_module(n_components=2)
    pca.fit(x_train)
    pca_components = pca.transform(x_train)
    pca.fit(x_test)
    pca_components_test= pca.transform(x_test)
    tsne = TSNE_unsupervise_module(n_components=2,random_state=1111)
    tsne_components = tsne.fit_transform(x_train)
    tsne_components_test = tsne.fit_transform(x_test)
    _, ((ax1, ax2), (ax3, ax4))  = plt.subplots(2, 2, figsize=(12, 12))
    ax1.scatter(pca_components[:, 0], pca_components[:, 1], c=color_train)
    ax1.set_title('PCA train')
    ax2.scatter(tsne_components[:, 0], tsne_components[:, 1], c=color_train)
    ax2.set_title('t-SNE train')
    ax3.scatter(pca_components_test[:, 0], pca_components_test[:, 1], c=color_test)
    ax3.set_title('PCA Test')
    ax4.scatter(tsne_components_test[:, 0], tsne_components_test[:, 1], c=color_test)
    ax4.set_title('t-SNE Test')
    plt.savefig('app/resources/pca_vs_tsne_unsupervised_module.png')

def PCA_Training_unsupervised_module(x_train,y_train,x_test,y_test):
    pca = PCA_unsupervised_module(n_components=2)
    pca.fit(x_train)
    x_train_pca = pca.transform(x_train)
    pca.fit(x_test)
    x_test_pca = pca.transform(x_test)
    model = LogisticRegression(random_state=1111)
    model.fit(x_train_pca, y_train)
    return model.score(x_test_pca, y_test)


def TSNE_Training_unsupervised_module(x_train,y_train,x_test,y_test):
    tsne = TSNE_unsupervise_module(n_components=2)
    x_train_tsne = tsne.fit_transform(x_train)
    x_test_tsne = tsne.fit_transform(x_test)
    model = LogisticRegression(random_state=1111)
    model.fit(x_train_tsne, y_train)
    return model.score(x_test_tsne, y_test)

def save_models(x_train,y_train,x_test,y_test):
    tsne = TSNE_Training_unsupervised_module(n_components=2,random_state=1111)
    x_train_tsne = tsne.fit_transform(x_train)
    model_tsne = LogisticRegression(random_state=1111)
    model_tsne.fit(x_train_tsne, y_train)
    with open('src/resources/trained_TSNE_model-0.1.0.pkl', 'wb') as file:
        pickle.dump(model_tsne, file)
    pca = PCA_unsupervised_module(n_components=2,random_state=1111)
    x_train_pca = pca.fit_transform(x_train)
    model_pca = LogisticRegression(random_state=1111)
    model_pca.fit(x_train_pca, y_train)
    with open('src/resources/trained_PCA_model-0.1.0.pkl', 'wb') as file:
        pickle.dump(model_tsne, file)

def save_models(x_train,y_train,x_test,y_test):
    tsne = TSNE_Training_unsupervised_module(n_components=2,random_state=1111)
    x_train_tsne = tsne.fit_transform(x_train)
    model_tsne = LogisticRegression(random_state=1111)
    model_tsne.fit(x_train_tsne, y_train)

    pca = PCA_unsupervised_module(n_components=2,random_state=1111)
    x_train_pca = pca.fit_transform(x_train)
    model_pca = LogisticRegression(random_state=1111)
    model_pca.fit(x_train_pca, y_train)
    
    with open('app/resources/trained_TSNE_model-0.1.0.pkl', 'wb') as file:
        pickle.dump(model_tsne, file)
    with open('app/resources/trained_PCA_model-0.1.0.pkl', 'wb') as file:
        pickle.dump(model_pca, file)
    with open('app/resources/PCA_scaler-0.1.0.pkl', 'wb') as file:
        pickle.dump(pca, file)
    with open('app/resources/TSNE_scaler-0.1.0.pkl', 'wb') as file:
        pickle.dump(tsne, file)
    with open('app/resources/TSNE_data-0.1.0.pkl', 'wb') as file:
        pickle.dump(x_train_tsne, file)