from fastapi import FastAPI
from app.modules.matrix import Matrix
from app.modules.Unsupervised import PCA_Training_scikit_unsupervised_module,TSNE_Training_unsupervised_module,plot_PCA_TSNE_unsupervised_module
from app.modules.Picture import Pictures
from fastapi.responses import FileResponse,HTMLResponse
from app.modules.scikit_learn_methods import load_mnist_dataset,Train_model_scikit_learn,PCA_Training_scikit_learn,TSNE_Training_scikit_learn,plot_PCA_TSNE_scikit

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


@app.post("/apply_SDV_MyImage")
async def apply_SDV_MyImage():
    pass


@app.get("/Train_mnist_model_scikit_learn")
async def Train_mnist_model_scikit_learn():
    x_train,y_train,x_test,y_test=load_mnist_dataset()
    score=Train_model_scikit_learn(x_train,y_train,x_test,y_test)
    return {'accuracy':score}


@app.get("/plot_PCA_TSNE_scikit_learn")
async def Train_mnist_model_local_unsupervised():
    plot_PCA_TSNE_scikit()
    return FileResponse('app/resources/pca_vs_tsne_scikit.png')


@app.get("/plot_PCA_TSNE_my_unsupervised_module")
async def plot_PCA_TSNE_my_unsupervised_module():
    plot_PCA_TSNE_unsupervised_module()
    return FileResponse('app/resources/pca_vs_tsne_my_unsupervised_module.png')
    
@app.get("/accuracy_PCA_TSNE_scikit_learn")
async def accuracy_PCA_TSNE_scikit_learn():
    x_train,y_train,x_test,y_test=load_mnist_dataset()
    normal_score=Train_model_scikit_learn(x_train,y_train,x_test,y_test)
    PCA_score=PCA_Training_scikit_learn(x_train,y_train,x_test,y_test)
    TSNE_score=TSNE_Training_scikit_learn(x_train,y_train,x_test,y_test)
    return {'normal_score':normal_score,'PCA_score':PCA_score,'TSNE_score':TSNE_score,}

@app.get("/accuracy_PCA_TSNE_my_unsupervised_module")
async def accuracy_PCA_TSNE_scikit_learn():
    x_train,y_train,x_test,y_test=load_mnist_dataset()
    normal_score=Train_model_scikit_learn(x_train,y_train,x_test,y_test)
    PCA_score=PCA_Training_scikit_unsupervised_module(x_train,y_train,x_test,y_test)
    TSNE_score=TSNE_Training_unsupervised_module(x_train,y_train,x_test,y_test)
    return {'normal_score':normal_score,'PCA_score':PCA_score,'TSNE_score':TSNE_score,}

@app.get("/technical_methods_improve_PCA")
async def technical_methods_improve_PCA():
    return {'method1':'this method'}


@app.get("/mathematical_principles_behind_UMAP")
async def mathematical_principles_behind_UMAP():
    return {'method1':'this method'}


@app.get("/mathematical_principles_behind_LDA")
async def mathematical_principles_behind_LDA():
    return {'method1':'this method'}