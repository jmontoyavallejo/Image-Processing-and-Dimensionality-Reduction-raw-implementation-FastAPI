from app.modules.matrix import Matrix
from app.modules.Unsupervised import PCA_Training_unsupervised_module,TSNE_Training_unsupervised_module,plot_PCA_TSNE_unsupervised_module,SVD_unsupervised_module
from app.modules.Picture import Pictures
from app.modules.scikit_learn_methods import load_mnist_dataset,Train_model_scikit_learn,PCA_Training_scikit_learn,TSNE_Training_scikit_learn,plot_PCA_TSNE_scikit

from fastapi import FastAPI
from fastapi.responses import FileResponse,HTMLResponse


app = FastAPI()


@app.get("/",response_class=HTMLResponse)
def read_root():
    return '''<html>
    <div style="color: black;box-shadow: 8px 8px 5px #444;padding:5em;border: 1px solid #333;text-align: center;margin: auto;background-image: linear-gradient(180deg, #fff, #ddd 40%, #ccc);width: 20em;border: 1px solid #333;font-size: 23px;"> 
    Hello! welcome <br> to Juan Pablo's dimensionality reduction repo, feel free to go to 
    <a href="http://localhost:8000/docs">Fastapi Docs</a>. 
    so you can interact with the repository  
    </div>
    </html>
    '''


@app.post("/matrix_calculations/")
async def calculate_matrix(cols:int, rows:int):
    my_matrix = Matrix(cols=cols, rows=rows)
    results={
    "values": my_matrix.values.tolist(),
    "rank": my_matrix.rank(),
    "trace": my_matrix.trace(),
    "determinant": my_matrix.determinant(),
    "inverse": my_matrix.inverse(),
    "eigenvalues_transpose": my_matrix.eigenvalues_transpose(),
    'eigenvalues response':'they are the same non zero values but in diferent order if the matrix is square'
    }
    return results


@app.get("/MyImage")
async def get_myimage():
    pictures = Pictures()
    pictures.save_my_image()
    return FileResponse('app/resources/my_image.png')


@app.get("/AverageImage")
async def get_Average_Image():
    pictures = Pictures()
    pictures.save_average_image()
    return FileResponse('app/resources/average_image.png')


@app.get("/compare_my_image_to_average_image")
async def compare_my_image_to_average_image():
    pass


@app.get("/apply_SDV_MyImage")
async def apply_SDV_MyImage():
    pictures = Pictures()
    SVD_unsupervised_module(pictures.cara0)
    return FileResponse('app/resources/SDV_image.png')


@app.get("/Train_mnist_model_scikit_learn")
async def Train_mnist_model_scikit_learn():
    x_train,y_train,x_test,y_test=load_mnist_dataset()
    score=Train_model_scikit_learn(x_train,y_train,x_test,y_test)
    return {'accuracy':score}


@app.get("/plot_PCA_TSNE_scikit_learn")
async def Train_mnist_model_local_unsupervised():
    plot_PCA_TSNE_scikit()
    return FileResponse('app/resources/pca_vs_tsne_scikit.png')


@app.get("/accuracy_PCA_TSNE_scikit_learn")
async def accuracy_PCA_TSNE_scikit_learn():
    x_train,y_train,x_test,y_test=load_mnist_dataset()
    normal_score=Train_model_scikit_learn(x_train,y_train,x_test,y_test)
    PCA_score=PCA_Training_scikit_learn(x_train,y_train,x_test,y_test)
    TSNE_score=TSNE_Training_scikit_learn(x_train,y_train,x_test,y_test)
    return {'normal_accuracy':normal_score,'PCA_accuracy':PCA_score,'TSNE_accuracy':TSNE_score}


@app.get("/plot_PCA_TSNE_my_unsupervised_module")
async def plot_PCA_TSNE_my_unsupervised_module():
    plot_PCA_TSNE_unsupervised_module()
    return FileResponse('app/resources/pca_vs_tsne_my_unsupervised_module.png')
    
@app.get("/accuracy_PCA_TSNE_my_unsupervised_module")
async def accuracy_PCA_TSNE_scikit_learn():
    x_train,y_train,x_test,y_test=load_mnist_dataset()
    normal_score=Train_model_scikit_learn(x_train,y_train,x_test,y_test)
    PCA_score=PCA_Training_unsupervised_module(x_train,y_train,x_test,y_test)
    TSNE_score=TSNE_Training_unsupervised_module(x_train,y_train,x_test,y_test)
    return {'normal_score':normal_score,'PCA_score':PCA_score,'TSNE_score':TSNE_score,}

@app.get("/technical_methods_improve_PCA")
async def technical_methods_improve_PCA():
    return {'method1':'this method includes making more solvers to the program like random_SDV'}


@app.get("/mathematical_principles_behind_UMAP")
async def mathematical_principles_behind_UMAP():
    return {'nerve theorem':'The nerve theorem is a mathematical principle that states that the topological structure of a space can be captured by a collection of overlapping sets, known as nerve covers. In the context of UMAP, the nerve theorem is used to construct a simplicial complex, which is a mathematical object that captures the topological structure of the data.',
            'Graph theory':'UMAP uses a graph-based approach to construct a low-dimensional embedding of high-dimensional data. Specifically, UMAP constructs a weighted k-nearest neighbor graph, where each data point is connected to its k nearest neighbors. The weights on these edges represent the degree of similarity between the connected points.',
            'Riemannian geometry':'UMAP constructs a low-dimensional embedding of the high-dimensional data by minimizing a cost function that is based on a Riemannian metric. The Riemannian metric is a mathematical concept from differential geometry that measures the curvature of a space. In the context of UMAP, the Riemannian metric is used to measure the distance between points in the low-dimensional space.',
            'Topological Data Analysis':'Topological Data Analysis (TDA) is a branch of mathematics that uses algebraic topology to study the topological structure of data. In the context of UMAP, TDA is used to construct a simplicial complex, which is a topological object that captures the topological structure of the data. Specifically, UMAP uses the Mapper algorithm, which is a TDA algorithm that constructs a simplicial complex based on a cover of overlapping sets.'}


@app.get("/mathematical_principles_behind_LDA")
async def mathematical_principles_behind_LDA():
    return {'Dirichlet distribution':'The Dirichlet distribution is a probability distribution over a simplex, which is a multi-dimensional space whose points all lie on the surface of a hyperplane. In the context of LDA, the Dirichlet distribution is used to model the distribution of topics over a set of documents. Specifically, each document is represented as a mixture of topics, and the Dirichlet distribution is used to model the probability distribution of these mixtures.',
            'Bayesian inference':'LDA is a Bayesian model, which means that it uses Bayes theorem to perform probabilistic inference. Bayes theorem is a mathematical formula that allows us to update our beliefs about a hypothesis based on new evidence. In the context of LDA, we use Bayes theorem to infer the underlying topic structure of a set of documents. Specifically, we start with a prior belief about the topic structure, and we update this belief based on the observed words in the documents.',
            'Matrix factorization':'LDA can be viewed as a form of matrix factorization, where we factorize a matrix of word frequencies into two matrices: one representing the distribution of topics over the words, and the other representing the distribution of documents over the topics. This factorization is achieved using Bayesian inference and optimization techniques.'}