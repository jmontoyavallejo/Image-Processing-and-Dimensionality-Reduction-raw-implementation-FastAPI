import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from keras.datasets import mnist
import matplotlib.pyplot as plt
import pickle 

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


def Train_model_scikit_learn(x_train,y_train,x_test,y_test):
    model = LogisticRegression(random_state=1111)
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

def plot_PCA_TSNE_scikit():
    x_train,y_train,x_test,y_test=load_mnist_dataset()
    color_train=['r' if i==0 else 'b' for i in y_train ]
    color_test = ['r' if i == 0 else 'b' for i in y_test]
    pca = PCA(n_components=2,random_state=1111)
    pca_components = pca.fit_transform(x_train)
    pca_components_test= pca.fit_transform(x_test)
    tsne = TSNE(n_components=2,random_state=1111)
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
    plt.savefig('app/resources/pca_vs_tsne_scikit.png')

def PCA_Training_scikit_learn(x_train,y_train,x_test,y_test):
    pca = PCA(n_components=2,random_state=1111)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    model = LogisticRegression(random_state=1111)
    model.fit(x_train_pca, y_train)
    return model.score(x_test_pca, y_test)


def TSNE_Training_scikit_learn(x_train,y_train,x_test,y_test):
    tsne = TSNE(n_components=2,random_state=1111)
    x_train_tsne = tsne.fit_transform(x_train)
    x_test_tsne = tsne.fit_transform(x_test)
    model = LogisticRegression(random_state=1111)
    model.fit(x_train_tsne, y_train)
    return model.score(x_test_tsne, y_test)
