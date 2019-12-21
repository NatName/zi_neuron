from __future__ import absolute_import, division, print_function, unicode_literals
import cv2
from random import shuffle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import models, layers
from keras.optimizers import *
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
train_data = 'C:/Users/NatName/PyCharmProjects/KeyboardProject/train'
test_data = 'C:/Users/NatName/PyCharmProjects/KeyboardProject/test'
users_array = []

def one_hot_label(img):
    label = img.split('.')
    if int(label[1]) not in users_array:
        users_array.append(int(label[1]))
    return [int(label[1])]


def train_data_with_label():
    train_images = []
    for i in tqdm(os.listdir(train_data)):
        path = os.path.join(train_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        train_images.append([np.array(img), one_hot_label(i)])
    shuffle(train_images)
    return train_images


def test_data_with_label():
    test_images = []
    for i in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        test_images.append([np.array(img), one_hot_label(i)])
    shuffle(test_images)
    return test_images


training_images = train_data_with_label()
testing_images = test_data_with_label()
print(training_images)
tr_img_data = np.array([i[0] for i in training_images]).reshape(-1, 28, 28, 1) / 255.0
for i in tr_img_data:
    for j in i:
        print(j, end=' ')
    print("\n################\n")
print(tr_img_data)
tr_lbl_data = np.array([i[1] for i in training_images])
tst_img_data = np.array([i[0] for i in testing_images]).reshape( -1,28, 28, 1) / 255.0
tst_lbl_data = np.array([i[1] for i in testing_images])


# Определим простую последовательную модель
# def create_model():
#     model = tf.keras.models.Sequential([
#         keras.layers.Dense(512, activation='relu', input_shape=(1600,)),
#         keras.layers.Dropout(0.2),
#         keras.layers.Dense(10, activation='softmax')
#     ])
#
#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#
#     return model
#
#
# # Создадим экземпляр базовой модели
# model = create_model()
model = Sequential()

# C1 Convolutional Layer
model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(28, 28, 1), padding='same'))

# S2 Pooling Layer
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))

# C3 Convolutional Layer
model.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))

# S4 Pooling Layer
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# C5 Fully Connected Convolutional Layer
model.add(layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))

#Flatten the CNN output so that we can connect it with fully connected layers
model.add(layers.Flatten())

# FC6 Fully Connected Layer
model.add(layers.Dense(84, activation='tanh'))

#Output Layer with softmax activation
model.add(layers.Dense(1, activation='softmax'))

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='SGD', metrics=['accuracy'])
# checkpoint_path = "training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# model.load_weights(checkpoint_path)
# # Создаем коллбек сохраняющий веса модели
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)
# Обучаем модель с новым коллбеком
model.fit(tr_img_data, tr_lbl_data, epochs=10, batch_size=128, validation_data=(tst_img_data, tst_lbl_data), verbose=1)

# Загрузим веса
# model.load_weights(checkpoint_path)
# Оцените модель
loss, acc = model.evaluate(tst_img_data, tst_lbl_data, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model.from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")
loss, acc = model.evaluate(tst_img_data, tst_lbl_data, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))
# # evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate(x, y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))


# model = Sequential()
#
# model.add(InputLayer(input_shape=[100, 75, 256]))
# model.add(Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu'))
# model.add(MaxPool2D(pool_size=5, padding='same'))
#
# model.add(Conv2D(filters=50, kernel_size=5, strides=1, padding='same', activation='relu'))
# model.add(MaxPool2D(pool_size=5, padding='same'))
#
# model.add(Conv2D(filters=80, kernel_size=5, strides=1, padding='same', activation='relu'))
# model.add(MaxPool2D(pool_size=5, padding='same'))
#
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(rate=0.5))
# model.add(Dense(2, activation='softmax'))
# optimizer = Adam(lr=1e-3)
#
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(x=to_categorical(tr_img_data), y= to_categorical(tr_lbl_data), epochs=10, batch_size=100)
# model.summary()
#
# fig = plt.figure(figsize=(14, 14))
# print(testing_images)
# for cnt, data in enumerate(testing_images):
#     y = fig.add_subplot(6, 5, cnt+1)
#     img = data[0]
#     data = img.reshape(1, 100, 75, 1)
#     model_out = model.predict(to_categorical(data))
#     if np.argmax(model_out) == 1:
#         str_label = 't'
#     else:
#         str_label = 'v'
#     y.imshow(img, cmap='gray')
#     plt.title(str_label)
#     y.axes.get_xaxis().set_visible(False)
#     y.axes.get_yaxis().set_visible(False)
#     plt.show()
