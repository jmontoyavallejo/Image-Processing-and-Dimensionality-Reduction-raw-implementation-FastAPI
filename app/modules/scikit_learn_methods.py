import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from keras.datasets import mnist
import matplotlib.pyplot as plt

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


def Train_model_scikit_learn(x_train,y_train,x_test,y_test):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

def plot_PCA_TSNE_scikit():
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
    plt.savefig('app/resources/pca_vs_tsne_scikit.png')

def PCA_Training_scikit_learn(x_train,y_train,x_test,y_test):
    pca = PCA(n_components=2)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    model = LogisticRegression()
    model.fit(x_train_pca, y_train)
    return model.score(x_test_pca, y_test)


def TSNE_Training_scikit_learn(x_train,y_train,x_test,y_test):
    tsne = TSNE(n_components=2)
    x_train_tsne = tsne.fit_transform(x_train)
    x_test_tsne = tsne.fit_transform(x_test)
    model = LogisticRegression()
    model.fit(x_train_tsne, y_train)
    return model.score(x_test_tsne, y_test)