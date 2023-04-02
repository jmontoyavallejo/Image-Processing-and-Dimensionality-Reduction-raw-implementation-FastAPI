from typing import Union

from fastapi import FastAPI
from app.modules.matrix import Matrix
from app.modules.Unsupervised import PCA,SVD,TSNE
from app.modules.Picture import Pictures
from fastapi.responses import FileResponse
app = FastAPI()



@app.get("/")
def read_root():
    return {"Hello": "World"}


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
def get_myimage():
    pictures = Pictures()
    pictures.save_my_image()
    return FileResponse('app/resources/my_image.png')

@app.get("/AverageImage")
def get_Average_Image():
    pictures = Pictures()
    pictures.save_average_image()
    return FileResponse('app/resources/average_image.png')

@app.post("/SDV_MyImage")
def get_Average_Image():
    pictures = Pictures()
    pictures.save_average_image()
    return FileResponse('app/resources/average_image.png')