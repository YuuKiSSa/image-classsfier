import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image
from keras import layers
from keras.preprocessing.image import ImageDataGenerator


def encode_onehot(pos, n_rows):
    y_onehot = [0] * 4

    y_onehot[pos] = 1
    y_onehots = [y_onehot] * n_rows

    return np.array(y_onehots)


def read_img_data(path, fruit):
    for file in os.listdir(path):
        if file[0] == '.':
            continue
        if fruit not in file:
            continue

        img = Image.open("{}/{}".format(path, file)).convert('RGB')
        img_resized = img.resize((224,224))

        data = np.array([np.asarray(img_resized)])

        try:
            x_train = np.concatenate((x_train, data))
        except:
            x_train = data

    return np.reshape(x_train, (-1, 224, 224, 3))


def prep_data(path):
    fruits = ['apple', 'banana', 'orange', 'mixed']
    
    for i in range(4):
        data = read_img_data(path, fruits[i])

        try:
            x = np.concatenate((x, data))
        except:
            x = data

        y_onehot = encode_onehot(i, data.shape[0])
        try:
            y = np.concatenate((y, y_onehot))
        except:
            y = y_onehot

    return x, y


def create_model():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units=128, activation='relu'))    

    model.add(layers.Dense(units=4, activation='softmax'))    

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def plot(hist):
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    ax[0].plot(hist.history['loss'])
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss Curve')

    ax[1].plot(hist.history['accuracy'])
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Accuracy Curve')

    plt.show()


model = create_model()

x_train, y_train = prep_data("C:/Users/98173/Desktop/GDipSA/Machine Learning/Fruits Classfier/train")

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
datagen.fit(x_train/255)

hist = model.fit(x_train/255, y_train, epochs=8)

plot(hist)

x_test, y_test = prep_data("C:/Users/98173/Desktop/GDipSA/Machine Learning/Fruits Classfier/test")

model.evaluate(x_test/255, y_test)
