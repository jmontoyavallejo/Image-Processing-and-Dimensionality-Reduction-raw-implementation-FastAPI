import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh,norm

class Pictures:
    def __init__(self):
        module_path = os.path.dirname(os.path.abspath(__file__))
        resources_path = os.path.join(module_path, '../resources/full_images')
        self.directory = resources_path
        self.images = self.load_images()
        self.cara0 = self.images[0]

    def load_images(self):
        images = []
        for filename in os.listdir(self.directory):
            if filename.endswith('.png'):
                filepath = os.path.join(self.directory, filename)
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                resized_image = cv2.resize(image, (256, 256))
                images.append(resized_image)
        return np.array(images)

    def save_my_image(self):
        png_image = cv2.imencode('.png', self.cara0)[1].tobytes()

        with open('app/resources/my_image.png', 'wb') as f:
            f.write(png_image)

    def save_average_image(self):
        avg_image = np.array(np.mean(self.images, axis=(0)), dtype=np.uint8)
        png_image = cv2.imencode('.png', avg_image)[1].tobytes()
        with open('app/resources/average_image.png', 'wb') as f:
            f.write(png_image)      

    def save_svd_image(self):
        def SVD_unsupervised_module(A):
            ev,V=eigh(A.T@A)
            u=[]
            for i in range(A.shape[1]):
                u.append(A@V[:,i]/norm(A@V[:,i]))
            U=np.array(u).T
            S=np.round(U.T@A@V,decimals=5)
            return U,np.diagonal(S),V.T
        U,s,VT=SVD_unsupervised_module(self.cara0)
        S=np.diag(s)
        rows = 3
        cols = 3
        fig, axs = plt.subplots(rows, cols, figsize=(10, 8))
        j = 0
        for i in (5, 20, 50,80, 100,150,190,210 ,256):
            img_aprox = U[:, :i] @ S[0:i, :i] @ VT[:i, :]
            resized_aprox = cv2.resize(img_aprox, (256//cols, 256//rows))
            row = j // cols
            col = j % cols
            axs[row, col].imshow(resized_aprox, cmap='gray')
            axs[row, col].axis('off')
            axs[row, col].set_title(f'valores singulares = {i}')
            j += 1
        plt.savefig('app/resources/SDV_image.png')

    def distance_average_my_image(self):
        avg_image = np.array(np.mean(self.images, axis=(0)), dtype=np.uint8)
        diff=self.cara0 -avg_image
        fro_dist = np.linalg.norm(diff, ord='fro')
        return fro_dist
    
 

