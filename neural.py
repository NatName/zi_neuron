from __future__ import absolute_import, division, print_function, unicode_literals
import cv2
from random import shuffle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
train_data = './../train'
test_data = './../test'
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
        img = cv2.resize(img, (40, 40))
        train_images.append([np.array(img), one_hot_label(i)])
    shuffle(train_images)
    return train_images


def test_data_with_label():
    test_images = []
    for i in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (40, 40))
        test_images.append([np.array(img), one_hot_label(i)])
    shuffle(test_images)
    return test_images


training_images = train_data_with_label()
testing_images = test_data_with_label()
print(training_images)
tr_img_data = np.array([i[0] for i in training_images]).reshape(-1, 40 * 40) / 255.0
tr_lbl_data = np.array([i[1] for i in training_images])
tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1, 40 * 40) / 255.0
tst_lbl_data = np.array([i[1] for i in testing_images])


# Определим простую последовательную модель
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(1600,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

print("1")
# Создадим экземпляр базовой модели
model = create_model()
print("2")
checkpoint_path = "./../training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
print("3")
model.load_weights(checkpoint_path)

# Создаем коллбек сохраняющий веса модели
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
# Обучаем модель с новым коллбеком
model.fit(tr_img_data,
          tr_lbl_data,
          epochs=10,
          validation_data=(tst_img_data, tst_lbl_data),
          callbacks=[cp_callback])  # Pass callback to training
# Загрузим веса
#model.load_weights(checkpoint_path)
# Оцените модель
loss, acc = model.evaluate(tst_img_data, tst_lbl_data, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))
# Распечатаем архитектуру модели
model.summary()
