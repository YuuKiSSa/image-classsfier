from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np


def read_img_data(path, fruit):
    files=[]
    for file in os.listdir(path):
        if file[0] == '.':
            continue
        if fruit not in file:
            break

        img = Image.open("{}/{}".format(path, file)).convert('RGB')
        img_resized = img.resize((224,224))

        data = np.array([np.asarray(img_resized)])

        try:
            files.append(file)
            x_train = np.concatenate((x_train, data))
        except:
            x_train = data
        
    return x_train.reshape((-1, 224, 224, 3)), files

x, files = read_img_data("C:/Users/98173/Desktop/GDipSA/Machine Learning/Fruits Classfier/train", "apple")
print(x.shape)
print(files)