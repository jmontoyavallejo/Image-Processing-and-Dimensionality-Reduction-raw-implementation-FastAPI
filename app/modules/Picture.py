import os
import cv2
import numpy as np

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


