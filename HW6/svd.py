from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def load_image():
    image = Image.open("./Alan_Turing.jpg")
    data = np.array(image)
    print(data.shape)
    return data

def svd():
    data = load_image()
    u, s, vh = np.linalg.svd(data)
    return u, s, vh

def compress():
    u, s, vh = svd()
    k = [2, 4, 8, 16, 32, 64, 128, 256]
    for i in k:
        A = u[:,:i]@np.diag(s[:i])@vh[:i,:]
        plt.imshow(A, cmap='gray')
        title = "Compress with k = " + str(i)
        plt.title(title)
        plt.show()

if __name__ == "__main__":
    compress()
