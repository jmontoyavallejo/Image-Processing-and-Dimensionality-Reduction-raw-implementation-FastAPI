from typing import Union

from fastapi import FastAPI
from app.modules.matrix import Matrix
from app.modules.Unsupervised import PCA,SVC,TSNE
from app.modules.Picture import Picture
from typing import List, Dict, Union
import numpy as np
from matplotlib import pyplot as plt
import io
import base64
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

@app.get("/line-plot")
def line_plot():
    # Crea los datos para la gráfica
    x = [1, 2, 3, 4, 5]
    y = [1, 4, 9, 16, 25]
    
    # Crea la gráfica
    fig, ax = plt.subplots()
    ax.plot(x, y)
    
    # Convierte la gráfica a una imagen
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    
    # Convierte la imagen a base64
    plot_data = base64.b64encode(img.getvalue()).decode()
    
    # Devuelve la imagen como respuesta
    return {"image": plot_data}