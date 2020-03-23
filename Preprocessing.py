from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import requests
from functools import lru_cache as cache

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.compat.v2.nn import relu6
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split
from tqdm import tqdm


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


def getConfMatrix(model, X_Test, y_true, to_categorical=False, y_pred=None):
    '''
    Создание матрицы ошибок (в %)
    :param model: tensorflow.python.keras.engine.training.Model. обученная tf.keras модель
    :param X_Test: numpy.array(dtype=numpy.float32). data to predict
    :param y_true: numpy.array(dtype=numpy.uint8) or list. true labels
    :param y_pred: numpy.array(dtype=numpy.uint8) or list. predict labels
    :param to_categorical: bool. если labels в one hot encoding
    '''
    if to_categorical:
        num_pics_test = np.unique(y_true, return_counts=True, axis=-2)[1]
    else:
        num_pics_test = np.unique(y_true, return_counts=True)[1]
    if y_pred:
        data = {'y_Actual': y_true,
                'y_Predicted': y_pred
                }
    else:
        data = {'y_Actual': y_true,
                'y_Predicted': [p.argmax() for p in model.predict(X_Test)]
                }
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'],
                                   df['y_Predicted'],
                                   rownames=['Actual'],
                                   colnames=['Predicted'],
                                   margins=True)
    print('======>')
    for row, countInClass in enumerate(num_pics_test):
        confusion_matrix.loc[row] = (confusion_matrix.loc[row] * 100) / countInClass
        confusion_matrix.loc[row] = np.around(confusion_matrix.loc[row], 2)
    return confusion_matrix


def collaps_fich_matrix(path, model_for_vis, in_shape=(128, 128)):
    '''
    Отражение матрицы признаков каждого выгодного слоя свёрточной сети.
    Схлапывание всех слоёв в 1.
    :param path: string. Путь к файлу изображения
    :param model_for_vis: tensorflow.python.keras.engine.training.Model. 'Обрезанная где-то' сеть
    :param in_shape: tuple. Необходимая размерность
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
    :param model_for_vis1: tensorflow.python.keras.engine.training.Model. 'Обрезанная где-то в середине архитектуры' сеть
    :param model_for_vis2: tensorflow.python.keras.engine.training.Model. 'Обрезанная к концу архитектуры' сеть
    :param path: string. Путь к файлу изображения
    :param t: string. Текст
    :param in_shape: tuple. Необходимая размерность
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


def new_smooth_label(categorical_labels, smooth_factor=0.2):
    '''
    Функция "распыляет" метки,
    переданные в формате onehotencoding
    :param categorical_labels: np.array(dtype=numpy.uint8) либо list. метки в формате onehotencoding
    :param smooth_factor: float. процент "размытия"
    '''
    new_y = []
    for y_ in categorical_labels:
        y = y_ * (1 - smooth_factor)
        y += (smooth_factor / categorical_labels.shape[-1])
        new_y.append(y)
    new_y = np.array(new_y, dtype=np.float16)
    return new_y


def imgs2array(main_dir, width=128, height=128, channels=3):
    '''
    Перевод изображений в numpy.array
    :param main_dir: string. каталог с подкаталогами(они же классы) изображений
    :param width: integer. нужная ширина
    :param height: integer. нужная высота
    :param channels: integer. каналы
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
    :param X_train: numpy.array(dtype=numpy.float32). Данные
    :param y_train: numpy.array(dtype=numpy.uint8) or list. Метки.
    :param trainValid: float. проценты раскусывания на Valid
    :param trainTest: float. проценты раскусывания на Test
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
    :param path: string. путь к файлу
    :param X_train: numpy.array(dtype=numpy.float32). Данные
    :param y_train:  numpy.array(dtype=numpy.uint8) or list. Метки
    :param compress: string. метод сжатия
    :param compress_opts: integer. степень сжатия
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
    :param path: string. путь к файлу
    :param rescale: bool. флаг для нормализации [-1 1]
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


def gen2pred(model, path, x, y, batch=1000):
    '''
    Генератор. Проходит по файлу, нормализует батч х,
    предсказывает через модель.
    Выдаёт предсказанные y и истинные y.
    :param model: tensorflow.python.keras.engine.training.Model. Модель в Keras
    :param path: string. Путь к .hdf5 файлу
    :param x: string. Ключ в .h5py к X_train
    :param y: string. Ключ в .h5py к y_train
    :param batch: integer. Размер батча
    '''
    inx = 0 # Индекс для контроля перемещения по .hdf5
    y_pred_all = [] # Список с распознанными метками классов
    y_true_all = [] # Список с истинными метками классов
    if batch == 0:
        batch = 1
    while True:
        with h5py.File(path, 'r') as f:
            x_ = f[x][inx:inx + batch]
            x_ = x_.astype(np.float32)
            x_ /= 128.
            x_ -= 1.

            y_pred = model.predict(x_)
            y_pred = [p.argmax() for p in y_pred]  # -> list
            y_true = f[y][inx:inx + batch]
            y_true = y_true.astype(np.uint8)  # -> np.uint8
            y_pred_all.extend(y_pred)  # -> list
            y_true_all.extend(y_true)  # -> list
            yield y_pred_all, y_true_all

            # Проверка. Если прошли весь файл, создаём исключение StopIteration
            if inx + batch == f[x].shape[0]:
                return False
            # Проверка. Если через следующую итерацию inx выбросит за пределы файла
            elif inx + 2 * batch > f[x].shape[0]:
                # Выполняем сдвиг inx
                inx += batch
                # Изменяем batch, чтобы не выпрыгнуть за пределы файла
                batch = f[x].shape[0] - inx
            else:
                inx += batch


def getConfUseGen(model, path, x, y, batch=1000):
    '''
    Создёт матрицу ошибок, используя генератор.
    :param model: tensorflow.python.keras.engine.training.Model. Модель в Keras
    :param path: string. Путь к .hdf5 файлу
    :param x: string. Ключ в .h5py к X_train
    :param y: string. Ключ в .h5py к y_train
    :param batch: integer. Размер батча
    '''
    y_pred, y_true = (0., 0.)
    gen = gen2pred(model, path, x, y, batch=batch)
    while True:
        try:
            y_pred, y_true = next(gen)
        except StopIteration:
            break
    confMtrx = getConfMatrix(model=model,
                             X_Test=None,
                             y_true=y_true,
                             to_categorical=False,
                             y_pred=y_pred)
    return confMtrx


def genFromH5(path, x, y, batch, num_classes, to_categorical=False):
    '''
    Генератор. Возвращает сбалансированный батч (x, y)
    x - нормализуется. x/=128. ; x-=1.
    :param path: string. Путь до .hdf5 файла
    :param x: string. Ключ в .h5py к X_train
    :param y: string. Ключ в .h5py к y_train
    :param batch: integer. Размер батча
    :param num_classes: integer. Количество классов
    :param to_categorical: bool. Нужен ли smooth_label
    '''
    # Балансировка батча. Целое число (x,y) в кажом классе
    bath_on_class = batch // num_classes
    # Создание словаря с использованными индексами лэйблов
    used_inxs = {key: [] for key in [inx for inx in range(num_classes)]}

    ###############Доработать|||||||||
    if batch != bath_on_class * num_classes:
        # Если при batch / num_classes есть остаток от деления
        # Узнать сколько не хватает
        add_inxs = batch - bath_on_class * num_classes
    ###############|||||||||||||||||||

    while True:
        with h5py.File(path, 'r') as f:
            inxs = []  # Массив с рандомными индексами. Используются при формировании батча

            ###############Доработать|||||||||
            if batch != bath_on_class * num_classes:
                rand_i = np.random.randint(f[y].shape[0], size=(add_inxs))
            ###############|||||||||||||||||||

            for class_ in range(num_classes):
                # Фильтрация лэйблов по классам
                # Если used_inxs не пустой, то избавляемся от использованных индексов
                if len(used_inxs[class_]) > 0:
                    # Создание маски по одному классу. Общая длинна y-вектора не меняется.
                    mask_class = f[y][:] == class_ # -> bool
                    # Присуждение False использованным индексам класса
                    mask_class[used_inxs[class_]] = False
                    # Получение индексов, где True
                    mask_class = np.where(mask_class)[0]
                    # Если при следующей итерации все индексы будут использованны,
                    # то обнуляем список.
                    if mask_class.shape[0] - bath_on_class < bath_on_class:
                        used_inxs[class_] = []
                else:
                    mask_class = f[y][:] == class_
                    mask_class = np.where(mask_class)[0]  # => тут индексы по классу[0] numpy
                # Выбор случайных индексов, в количестве  bath_on_class
                rand_inxs = np.random.choice(mask_class, bath_on_class)

                # Добавление выбранных индексов от каждого класса в конец списка inxs
                # и по ключу в словарь used_inxs
                inxs.extend(rand_inxs)
                used_inxs[class_] += list(rand_inxs)  ################

            ###############Доработать|||||||||
            if batch != bath_on_class * num_classes:
                # Если при batch / num_classes есть остаток от деления
                # Узнать сколько не хватает
                add_inxs = batch - bath_on_class * num_classes
                # Выбираем рандомные индексы
                rand_labels = np.random.randint(f[y].shape[0], size=(add_inxs))
                # Добавляем рандомные индексы
                inxs.append(rand_labels)
            ###############|||||||||||||||||||

            # Перевод в np.array,чтобы сделать .shuffle
            inxs = np.array(inxs, dtype='int64')
            np.random.shuffle(inxs)

            # Собираем и скалируем x_batch
            x_batch = [f[x][inx] for inx in inxs]
            x_batch = np.array(x_batch, dtype=np.float32)
            x_batch /= 128.
            x_batch -= 1.

            # Собираем y_bath
            y_batch = [f[y][inx] for inx in inxs]
            y_batch = np.array(y_batch, dtype=np.uint8)
            if to_categorical:
                # Если установлен флаг, то перевод в onehotencoding и размытие меток
                y_batch = utils.to_categorical(y_batch, num_classes, dtype=np.uint8)
                y_batch = new_smooth_label(y_batch)

        yield x_batch, y_batch


def genFromRAM(x, y, batch, num_classes, to_categorical=False):
    '''
    Генератор. Возвращает сбалансированный батч (x, y)
    x - нормализуется. x/=128. ; x-=1.
    :param x: string. Ключ в .h5py к X_train
    :param y: string. Ключ в .h5py к y_train
    :param batch: integer. Размер батча
    :param num_classes: integer. Количество классов
    :param to_categorical: bool. Нужен ли smooth_label
    '''
    # Балансировка батча. Целое число (x,y) в кажом классе
    bath_on_class = batch // num_classes
    # Создание словаря с использованными индексами лэйблов
    used_inxs = {key: [] for key in [inx for inx in range(num_classes)]}

    ###############Доработать|||||||||
    if batch != bath_on_class * num_classes:
        # Если при batch / num_classes есть остаток от деления
        # Узнать сколько не хватает
        add_inxs = batch - bath_on_class * num_classes
    ###############|||||||||||||||||||

    while True:
        inxs = []  # Массив с рандомными индексами. Используются при формировании батча

        ###############Доработать|||||||||
        if batch != bath_on_class * num_classes:
            rand_i = np.random.randint(x.shape[0], size=(add_inxs))
        ###############|||||||||||||||||||

        for class_ in range(num_classes):
            # Фильтрация лэйблов по классам
            # Если used_inxs не пустой, то избавляемся от использованных индексов
            if len(used_inxs[class_]) > 0:
                # Создание маски по одному классу. Общая длинна y-вектора не меняется.
                mask_class = y == class_ # -> bool
                # Присуждение False использованным индексам класса
                mask_class[used_inxs[class_]] = False
                # Получение индексов, где True
                mask_class = np.where(mask_class)[0]
                # Если при следующей итерации все индексы будут использованны,
                # то обнуляем список.
                if mask_class.shape[0] - bath_on_class < bath_on_class:
                    used_inxs[class_] = []
            else:
                mask_class = y == class_
                mask_class = np.where(mask_class)[0]  # => тут индексы по классу[0] numpy
            # Выбор случайных индексов, в количестве  bath_on_class
            rand_inxs = np.random.choice(mask_class, bath_on_class)
            # Добавление выбранных индексов от каждого класса в конец списка inxs
            # и по ключу в словарь used_inxs
            inxs.extend(rand_inxs)
            used_inxs[class_] += list(rand_inxs)  ################

        ###############Доработать|||||||||
        if batch != bath_on_class * num_classes:
            # Если при batch / num_classes есть остаток от деления
            # Узнать сколько не хватает
            add_inxs = batch - bath_on_class * num_classes
            # Выбираем рандомные индексы
            rand_labels = np.random.randint(y.shape[0], size=(add_inxs))
            # Добавляем рандомные индексы
            inxs.append(rand_labels)
        ###############|||||||||||||||||||

        # Перевод в np.array,чтобы сделать .shuffle
        inxs = np.array(inxs, dtype='int64')
        np.random.shuffle(inxs)
        # Собираем и скалируем x_batch
        x_batch = [x[inx] for inx in inxs]
        x_batch = np.array(x_batch, dtype=np.float32)
        x_batch /= 128.
        x_batch -= 1.

        # Собираем y_bath
        y_batch = [y[inx] for inx in inxs]
        y_batch = np.array(y_batch, dtype=np.uint8)
        if to_categorical:
            # Если установлен флаг, то перевод в one-hot-encoding и размытие меток
            y_batch = utils.to_categorical(y_batch, num_classes, dtype=np.uint8)
            y_batch = new_smooth_label(y_batch)
        yield x_batch, y_batch


def lr_scheduler(epoch):
    '''
    Планировщик скорости обучения.
    Возвращает изменённый LearningRate (lr).
    Коэфициенты выбраны эперическим путём.
    [1-23]: lr -3
    [24-39]: lr -4
    [40-:]: lr-5
    :param epoch: integer. номер эпохи
    :return: numpy.float32
    '''
    decay_rate = .7
    decay_step = 1.5
    if epoch < 11:
        epoch_ = epoch + 20
        lr = pow(decay_rate, np.floor(epoch_ / decay_step))
        return lr
    elif epoch < 16:
        lr = 1e-3
        return lr
    elif epoch < 25:
        epoch_ = epoch + 15
        lr = pow(decay_rate, np.floor(epoch_ / decay_step))
        return lr
    elif epoch < 31:
        lr = 1e-4
        return lr
    else:
        lr = 1e-5
        return lr


def simple_lr_scheduler(epoch):
    '''
    Планировщик скорости обучения.
    Возвращает изменённый LearningRate (lr).
    Коэфициенты выбраны эперическим путём.
    [1-23]: lr -3
    [24-39]: lr -4
    [40-:]: lr-5
    :param epoch: integer. номер эпохи
    :return: numpy.float32
    '''
    decay_rate = .7
    decay_step = 1.5
    if epoch < 20:
        epoch_ = epoch + 20
        lr = 1e-3
        return lr
    # elif epoch < 40:
    #     lr = 1e-4
    #     return lr
    else:
        lr = 1e-4
        return lr


def one_epoch_confmtrx(epoch, logs):
    '''
    X_test, y_test должны быть нормализованны и храниться в RAM.
    NAME_APP, API_KEY должны быть объявлены.
    Создаёт матрицу ошибок X_test, y_test.
    Передаёт POST запрос на сервер IFTTT.
    :param epoch: integer. номер эпохи
    :param logs: dict. логи входной эпохи
    '''
    # if (epoch+1) % 10 == 0:
    #     clear_output()
    mtrx = getConfMatrix(model, X_test, y_test)
    print(mtrx)
    conf_values = {i: mtrx.values[i][i] for i in range(mtrx.values.shape[-1]-1)}

    acc = logs.get('acc') * 100
    acc = np.around(acc, 2)
    val_acc = logs.get('val_acc') * 100
    val_acc = np.around(val_acc, 2)
    merge_acc = f'{acc}/{val_acc}'

    inf = {'value1': epoch+1, 'value2': merge_acc, 'value3': conf_values}
    ifttt_url = f'https://maker.ifttt.com/trigger/{NAME_APP}/with/key/{API_KEY}'
    requests.post(ifttt_url, json = inf)


def createImgLinks(main_dir, need_global_path=True):
    '''
    Создание pandas dataframe c данными и ссылками на файлы.
    Столбцы:
    - NAME. Имя файла.
    - LABEL. Метка класса по каталогу (1 каталог - 1 класс)
    - PATH. Абсолютный путь к файлу.
    :param main_dir: string. абсолютный путь к дирректории с изображениями, сортированными по каталогам (1 каталог - 1 класс)
    :param need_global_path: bool. создание абсолютного пути к файлу. Пример: main_dir\your_class_dir\your_image.jpg
    :return: pandas.DataFrame
    '''
    db = pd.DataFrame(columns=['LABEL', 'NAME'])
    imgs = []
    labels = []
    for dir_ in os.listdir(main_dir):
        child_dir = os.path.join(main_dir, dir_)
        imgs.extend([img for img in os.listdir(child_dir)])
        labels.extend([int(dir_) for i in os.listdir(child_dir)])
    db['NAME'] = imgs
    db['LABEL'] = labels
    db['LABEL'] = db['LABEL'].astype('uint8')
    if need_global_path:
        db['PATH'] = db.apply(lambda row: os.path.join(main_dir, os.path.join(str(row['LABEL']), row['NAME'])), axis=1)
    return db


def Db2TrainValidTest(db, trainValidSize=0.2, trainTestSize=0.05, shuffle=True, random_state=None):
    '''
    Перемешивание данных и создание train, valid, test dataframes с использованием библиотеки sklearn
    :param db: pandas.DataFrame. dataframe c данными и ссылками на файлы
    :param trainValidSize: float. размер val выборки (от 0. до 1.)
    :param trainTestSize: float. размер test выборки (от 0. до 1.)
    :param shuffle: bool. перемешивать данные
    :param random_state: integer. фиксировать случайные значения
    :return: pandas.DataFrame
    '''
    train_db, valid_db = train_test_split(db, test_size=trainValidSize,
                                          shuffle=shuffle, random_state=random_state)
    train_db, test_db = train_test_split(train_db, test_size=trainTestSize,
                                         shuffle=shuffle, random_state=random_state)
    all_ = []
    train_counts = train_db.groupby('LABEL').count().values
    all_.append(train_counts[:, 0])

    val_counts = valid_db.groupby('LABEL').count().values
    all_.append(val_counts[:, 0])

    test_counts = test_db.groupby('LABEL').count().values
    all_.append(test_counts[:, 0])
    _ = [print(i) for i in all_]
    return train_db, valid_db, test_db


def createZeros(train_db, valid_db, test_db, dim=(128, 128, 3)):
    '''
    Резервирование помяти под массивы данных (np.uint8) train, valid, test.
    (на основе train, valid, test dataframes)
    np.uint8 - значения 0 - 255.
    :param train_db: pandas.DataFrame. pandas dataframe c ссылками на train выборку
    :param valid_db: pandas.DataFrame. pandas dataframe c ссылками на valid выборку
    :param test_db: pandas.DataFrame. pandas dataframe c ссылками на test выборку
    :param dim: tuple. размерность данных
    :return: numpy.zeros(dtype=numpy.uint8)
    '''
    h, w, c = dim

    X_train = np.zeros((train_db.shape[0], h, w, c), dtype=np.uint8)
    y_train = np.zeros((train_db.shape[0]), dtype=np.uint8)

    X_val = np.zeros((valid_db.shape[0], h, w, c), dtype=np.uint8)
    y_val = np.zeros((valid_db.shape[0]), dtype=np.uint8)

    X_test = np.zeros((test_db.shape[0], h, w, c), dtype=np.uint8)
    y_test = np.zeros((test_db.shape[0]), dtype=np.uint8)

    print(f'X_train: {round(sys.getsizeof(X_train) * 1e-6, 2)} Mb')
    print(f'y_train: {round(sys.getsizeof(y_train) * 1e-6, 2)} Mb')
    print(f'X_val:   {round(sys.getsizeof(X_val) * 1e-6, 2)} Mb')
    print(f'y_val:   {round(sys.getsizeof(y_val) * 1e-6, 2)} Mb')
    print(f'X_test:  {round(sys.getsizeof(X_test) * 1e-6, 2)} Mb')
    print(f'y_test:  {round(sys.getsizeof(y_test) * 1e-6, 2)} Mb')
    return X_train, X_val, X_test, y_train, y_val, y_test


def fillZeros(X, y, link_db):
    '''
    Наполнение пустых train, valid, test массивов данных
    :param X: numpy.array(dtype=numpy.uint8). пустой массив данных под образцы
    :param y: numpy.array(dtype=numpy.uint8). пустой массив данных под метки
    :param link_db: pandas.DataFrame. pandas dataframe c ссылками
    :return: numpy.array(dtype=numpy.uint8), numpy.array(dtype=numpy.uint8)
    '''
    index = 0
    for (class_, _, path2img) in tqdm(link_db.values):
        try:
            arr = Image.open(path2img, 'r')
            arr = np.array(arr, dtype=np.uint8)
            X[index] = arr
            y[index] = class_
            index += 1
        except:
            continue
    return X[:index], y[:index]
