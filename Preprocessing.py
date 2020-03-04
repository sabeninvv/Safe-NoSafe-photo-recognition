from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
from functools import lru_cache as cache

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.compat.v2.nn import relu6
from tensorflow.keras.models import load_model, model_from_json
from sklearn.model_selection import train_test_split


def reCreate_model(path_h5, path_json=None):
    '''
    Загрузка tf.keras модели в формате .h5 / .json
    '''
    if path_json:
        with open(path_json, 'r') as json_file:
            model = json_file.read()
            model = model_from_json(model)
        model.load_weights(path_h5)
    else:
        model = load_model(path_h5,
                           custom_objects={'relu6': relu6}
                           )
    return model


@cache(maxsize=None)
def crop2img2arr(path, in_shape=(128, 128), crop=False, center=False, scale=True):
    '''
    Перевод изображения в numpy.array
    :param path: путь к изображению
    :param in_shape: размер на выходе
    :param crop: добавление пустой области к меньшему размеру, чтобы получить квадрат
    :param center: нормализация центрорование
    :param scale:нормализация -1 +1
    '''
    img = Image.open(path, 'r')
    if crop:
        if img.size[0] >= img.size[1]:
            img = img.crop((0, 0, img.size[0], img.size[0]))
        else:
            img = img.crop((0, 0, img.size[1], img.size[1]))
    img = img.resize(in_shape)
    img = img.convert('RGB')
    if scale:
        if center:
            imgarr = np.array(img, dtype='float32')
            imgarr = imgarr - imgarr.mean()
            imgarr = imgarr / max(imgarr.max(), abs(imgarr.min()))
        else:
            imgarr = np.array(img, dtype='float32')
            imgarr /= 128.
            imgarr -= 1.
    else:
        imgarr = np.array(img, dtype='uint8')
    return imgarr


def getConfMatrix(model, x, y, to_categorical=False):
    '''
    Создание матрицы ошибок
    :param model: обученная tf.keras модель
    :param x: data to predict
    :param y: true labels
    #y_Predicted: predict labels
    :param to_categorical: если labels в one hot encoding
    '''
    if to_categorical:
        num_pics_test = np.unique(y, return_counts=True, axis=-2)[1]
    else:
        num_pics_test = np.unique(y, return_counts=True)[1]
    data = {'y_Actual': y,
            'y_Predicted': [p.argmax() for p in model.predict(x)]
            }
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'],
                                   df['y_Predicted'],
                                   rownames=['Actual'],
                                   colnames=['Predicted'],
                                   margins=True)
    print('====>')
    for row, countInClass in enumerate(num_pics_test):
        confusion_matrix.loc[row] = (confusion_matrix.loc[row] * 100) / countInClass
        confusion_matrix.loc[row] = np.around(confusion_matrix.loc[row], 3)
    return confusion_matrix


def conf_matrix_fitgen(model, test_generator, batch_size):
    Y_pred = model.predict_generator(test_generator, test_generator.samples // batch_size + 1)
    y_pred = np.argmax(Y_pred, axis=1)
    flag0, flag1, flag2 = (0, 0, 0)
    for i in test_generator.labels.astype('int32'):
        if i == 0:
            flag0 += 1
        elif i == 1:
            flag1 += 1
        else:
            flag2 += 1
    num_pics_test = [flag0, flag1, flag2]

    data = {'y_Actual': test_generator.labels.astype('int32'),
            'y_Predicted': y_pred
            }
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'],
                                   df['y_Predicted'],
                                   rownames=['Actual'],
                                   colnames=['Predicted'],
                                   margins=True)
    print('====>')
    for row, countInClass in enumerate(num_pics_test):
        confusion_matrix.loc[row] = (confusion_matrix.loc[row] * 100) / countInClass
        confusion_matrix.loc[row] = np.around(confusion_matrix.loc[row], 3)
    return confusion_matrix


def collaps_fich_matrix(path, model_for_vis, in_shape=(128, 128)):
    '''
    Отражение матрицы признаков каждого выгодного слоя свёрточной сети.
    Схлапывание всех слоёв в 1.
    :param path: Путь к файлу изображения
    :param model_for_vis: 'Обрезанная где-то' сеть
    :param in_shape: Необходимая размерность
    '''
    arr = crop2img2arr(path, in_shape=in_shape)
    img2pred = np.expand_dims(arr, axis=0)
    activation = model_for_vis.predict(img2pred)

    images_per_row = 1
    size = activation.shape[1]
    n_cols = 1

    display_grid = np.zeros((n_cols * size, images_per_row * size))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = activation[0, :, :, col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size,
                        row * size: (row + 1) * size
                        ] = (display_grid[col * size: (col + 1) * size,
                             row * size: (row + 1) * size] + channel_image) / 2
    return display_grid


def compare_images(model_for_vis, path, t):
    plt.figure(figsize=(14, 6))

    plt.subplot(121)
    img = Image.open(path).resize((200, 200))
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(122)
    display_grid = collaps_fich_matrix(path, model_for_vis)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='bwr')
    plt.axis("off")
    plt.title("Predict: " + str(t), fontsize=18)


def compare_images2(model_for_vis1, model_for_vis2, path, t, in_shape=(128, 128)):
    '''
    Визуализация матрицы признаков свёрточной сети
    :param model_for_vis1: 'Обрезанная где-то в середине архитектуры' сеть
    :param model_for_vis2: 'Обрезанная к концу архитектуры' сеть
    :param path: Путь к файлу изображения
    :param t: Текст
    :param in_shape: Необходимая размерность
    '''
    plt.figure(figsize=(21, 6))

    plt.subplot(131)
    img = Image.open(path).resize((200, 200))
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(132)
    display_grid = collaps_fich_matrix(path, model_for_vis1, in_shape=in_shape)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='bwr')
    plt.axis("off")
    plt.title("Predict: " + str(t), fontsize=18)

    plt.subplot(133)
    display_grid = collaps_fich_matrix(path, model_for_vis2, in_shape=in_shape)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='bwr')
    plt.axis("off")


def new_smooth_label(y_to_categorical, smooth_factor1=0.2, smooth_factor2=0.1):
    new_y = []
    for y_ in y_to_categorical:
        if np.argmax(y_) == 2:
            y = y_ * (1 - smooth_factor1)
            y += smooth_factor1 / 3
            new_y.append(y)
        else:
            y = y_ * (1 - smooth_factor2)
            y += smooth_factor2 / 3
            new_y.append(y)
    new_y = np.array(new_y)
    return new_y


def imgs2array(main_dir, width=128, height=128, channels=3):
    '''
    Перевод изображений в numpy.array
    :param main_dir: каталог с подкаталогами(они же классы) изображений
    :param width: нужная ширина
    :param height: нужная высота
    :param channels: каналы
    '''
    nums = 0
    for class_ in os.listdir(main_dir):
        class_dir = os.path.join(main_dir, class_)
        num_files = len(os.listdir(class_dir))
        nums += num_files
        print(f'Дирректория {class_}, {num_files} шт. файлов')
    print(f'В директориях: {os.listdir(main_dir)} , всего файлов: {nums} ')
    X_train = np.zeros((nums, width, height, channels), dtype=np.float32)
    y_train = np.zeros((nums), dtype=np.uint8)

    print('Перевод файлов в numpy array.')
    index = 0
    for class_ in os.listdir(main_dir):
        num = 0
        class_dir = os.path.join(main_dir, class_)
        start = time.time()
        for name_img in os.listdir(class_dir):
            path2img = os.path.join(class_dir, name_img)
            try:
                X_train[index] = crop2img2arr(path2img, in_shape=(width, height), crop=False)
                y_train[index] = int(class_)
                index += 1
                num += 1
            except:
                print(sys.exc_info())
                continue
        print(
            f'Обработан класс: {class_}. Время на обработку: {np.around(time.time() - start, 2)} сек. В классе изображений: {num}')
    print('Всего изображений: ', index)
    return X_train[:index], y_train[:index]


def getTrainValidTest(X_train, y_train, trainValid=0.2, trainTest=0.05, shuffle=True):
    '''
    Раскусывание Train.
    Создание Train, Valid, Test выборок
    :param trainValid: проценты раскусывания на Valid
    :param trainTestЖ проценты раскусывания на Test
    '''
    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      shuffle=shuffle,
                                                      test_size=trainValid)
    X_train, X_test, y_train, y_test = train_test_split(X_train,
                                                        y_train,
                                                        shuffle=shuffle,
                                                        test_size=trainTest)
    return X_train, X_val, X_test, y_train, y_val, y_test


def saveH5(path, X_train, y_train, compress=None, compress_opts=None):
    '''
    Раскусывание Train.
    Создание Train, Valid, Test выборок.
    Запись в .hdf5 файл
    :param path: путь к файлу
    :param X_train: Данные
    :param y_train: Метки
    :param compress: метод сжатия
    :param compress_opts: степень сжатия
    '''
    X_train, X_val, X_test, y_train, y_val, y_test = getTrainValidTest(y_train,
                                                                       X_train)
    with h5py.File(path, 'w') as f:
        f.create_dataset('X_train',
                         np.shape(X_train),
                         h5py.h5t.STD_U8BE,
                         data=X_train,
                         compression=compress,
                         compression_opts=compress_opts)
        f.create_dataset('y_train',
                         np.shape(y_train),
                         h5py.h5t.STD_U8BE,
                         data=y_train,
                         compression=compress,
                         compression_opts=compress_opts)
        f.create_dataset('X_val',
                         np.shape(X_val),
                         h5py.h5t.STD_U8BE,
                         data=X_val,
                         compression=compress,
                         compression_opts=compress_opts)
        f.create_dataset('y_val',
                         np.shape(y_val),
                         h5py.h5t.STD_U8BE,
                         data=y_val,
                         compression=compress,
                         compression_opts=compress_opts)
        f.create_dataset('X_test',
                         np.shape(X_test),
                         h5py.h5t.STD_U8BE,
                         data=X_test,
                         compression=compress,
                         compression_opts=compress_opts)
        f.create_dataset("y_test",
                         np.shape(y_test),
                         h5py.h5t.STD_U8BE,
                         data=y_test,
                         compression=compress,
                         compression_opts=compress_opts)


def loadH5(path, rescale=False):
    '''
    Загрузка .hdf5 файла
    Создание Train, Valid, Test выборок.
    :param path: путь к файлу
    :param rescale: нужно ли нормировать до [-1 1]
    '''
    def helpScale(h5py_File, name):
        x = h5py_File[name][:].astype(np.float32)
        x /= 128.
        x -= 1.
        return x

    with h5py.File(path, 'r') as f:
        if rescale:
            X_train = helpScale(f, name='X_train')
            X_val = helpScale(f, name='X_val')
            X_test = helpScale(f, name='X_test')
        else:
            X_train = f['X_train'][:]
            X_val = f['X_val'][:]
            X_test = f['X_test'][:]
        y_train = f['y_train'][:]
        y_val = f['y_val'][:]
        y_test = f['y_test'][:]
    return X_train, X_val, X_test, y_train, y_val, y_test
