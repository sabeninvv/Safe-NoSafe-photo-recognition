from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import random
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.utils.data_utils import Sequence


def smooth_label(categorical_labels, smooth_factor=0.2):
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


def genFromH5(path, x, y, batch, num_classes, singly=False, to_categorical=False, smooth=False):
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
    if singly:
        bath_on_class = batch // num_classes
        used_inxs = {key: [] for key in [inx for inx in range(num_classes)]}
    else:
        assert batch % num_classes == 0, ('Введите batch, кратный num_classes')
        bath_on_class = batch // num_classes
        used_inxs = {key: [] for key in [inx for inx in range(num_classes)]}

    while True:
        with h5py.File(path, 'r') as f:
            inxs = []
            for class_ in range(num_classes):
                if len(used_inxs[class_]) > 0:
                    mask_class = f[y][:] == class_  # -> bool
                    mask_class[used_inxs[class_]] = False
                    mask_class = np.where(mask_class)[0]
                    if mask_class.shape[0] - bath_on_class < bath_on_class:
                        used_inxs[class_] = []
                else:
                    mask_class = f[y][:] == class_
                    mask_class = np.where(mask_class)[0]  # => тут индексы по классу[0] numpy
                rand_inxs = np.random.choice(mask_class, bath_on_class)
                inxs.extend(rand_inxs)
                used_inxs[class_] += list(rand_inxs)

            inxs = np.array(inxs, dtype='int64')
            np.random.shuffle(inxs)
            x_batch = [f[x][inx] for inx in inxs]
            x_batch = np.array(x_batch, dtype=np.float32)
            x_batch /= 128.
            x_batch -= 1.

            y_batch = [f[y][inx] for inx in inxs]
            y_batch = np.array(y_batch, dtype=np.uint8)
            if to_categorical:
                y_batch = to_categorical(y_batch, num_classes, dtype=np.uint8)
                if smooth:
                    y_batch = smooth_label(y_batch)
        yield x_batch, y_batch


def genFromRAM(x, y, batch, num_classes, singly=False, to_categorical=False, smooth=False):
    '''
    Генератор. Возвращает сбалансированный батч (x, y)
    x - нормализуется. x/=128. ; x-=1.
    :param x: string. Ключ в .h5py к X_train
    :param y: string. Ключ в .h5py к y_train
    :param batch: integer. Размер батча
    :param num_classes: integer. Количество классов
    :param to_categorical: bool. Нужен ли smooth_label
    '''
    if singly:
        bath_on_class = batch // num_classes
        used_inxs = {key: [] for key in [inx for inx in range(num_classes)]}
    else:
        assert batch % num_classes == 0, ('Введите batch, кратный num_classes')
        bath_on_class = batch // num_classes
        used_inxs = {key: [] for key in [inx for inx in range(num_classes)]}

    while True:
        inxs = []
        for class_ in range(num_classes):
            if len(used_inxs[class_]) > 0:
                mask_class = y == class_  # -> bool
                mask_class[used_inxs[class_]] = False
                mask_class = np.where(mask_class)[0]
                if mask_class.shape[0] - bath_on_class < bath_on_class:
                    used_inxs[class_] = []
            else:
                mask_class = y == class_
                mask_class = np.where(mask_class)[0]
            rand_inxs = np.random.choice(mask_class, bath_on_class)
            inxs.extend(rand_inxs)
            used_inxs[class_] += list(rand_inxs)

        inxs = np.array(inxs, dtype='int64')
        np.random.shuffle(inxs)
        x_batch = [x[inx] for inx in inxs]
        x_batch = np.array(x_batch, dtype=np.float32)
        x_batch /= 128.
        x_batch -= 1.

        y_batch = [y[inx] for inx in inxs]
        y_batch = np.array(y_batch, dtype=np.uint8)
        if to_categorical:
            y_batch = to_categorical(y_batch, num_classes, dtype=np.uint8)
            if smooth:
                y_batch = smooth_label(y_batch)
        yield x_batch, y_batch

class MainGen:
    def __init__(self, path_to_dir, name_sample, dim, batch, trainValidSize=0.2, trainTestSize=0.05, shuffle=True,
                 random_state=41, to_categorical=False, smooth=False, resize=False,
                 interpolation=cv2.INTER_CUBIC, y_batch_dtype=np.uint8, up=1, column_y_labels='CLASS',
                 save_distortion_in_preproc=False):
        self.path_to_dir = path_to_dir
        assert name_sample in ['TRAIN', 'VALID', 'TEST'], ('Используйте "TRAIN", "VALID", "TEST"')
        self.name_sample = name_sample
        self.dim = dim
        self.interpolation = interpolation
        self.y_batch_dtype = y_batch_dtype
        self.column_y_labels = column_y_labels
        self.up = up
        self.save_distortion_in_preproc = save_distortion_in_preproc
        if save_distortion_in_preproc:
            self.resize = True
        else:
            self.resize = resize
        self.make_db(trainValidSize=trainValidSize, trainTestSize=trainTestSize, shuffle=shuffle,
                     random_state=random_state)
        self.get_constants()
        self.batch = batch
        self.to_categorical = to_categorical
        self.smooth = smooth


    def make_db(self, trainValidSize, trainTestSize, shuffle, random_state):
        paths_to_imgs = []
        y_all = []
        for dir_name in os.listdir(self.path_to_dir):
            child_dir = os.path.join(self.path_to_dir, dir_name)
            if os.path.isdir(child_dir) and dir_name.isdigit():
                for img_name in os.listdir(child_dir):
                    path_to_img = os.path.join(child_dir, img_name)
                    if os.path.isfile(path_to_img):
                        paths_to_imgs.append(path_to_img)
                        y_all.append(int(dir_name))
        db = pd.DataFrame(data={'PATH': paths_to_imgs,
                                'CLASS': y_all})

        train_db, valid_db = train_test_split(db, test_size=trainValidSize,
                                              shuffle=shuffle, random_state=random_state)
        train_db, test_db = train_test_split(train_db, test_size=trainTestSize,
                                             shuffle=shuffle, random_state=random_state)
        if self.name_sample == 'TRAIN':
            self.db_to_gen = train_db.reset_index(drop=True)
        elif self.name_sample == 'VALID':
            self.db_to_gen = valid_db.reset_index(drop=True)
        else:
            self.db_to_gen = test_db.reset_index(drop=True)

    def get_constants(self):
        self.db_to_gen['PropabilityNud'] = 0.
        self.db_to_gen['PropabilityNud'] = self.db_to_gen['PropabilityNud'].astype('float16')
        self.num_classes = self.db_to_gen['CLASS'].unique().size
        self.max_size = 0
        for class_ in self.db_to_gen['CLASS'].unique():
            if self.db_to_gen[self.db_to_gen['CLASS'] == class_].shape[0] > self.max_size:
                self.max_size = self.db_to_gen[self.db_to_gen['CLASS'] == class_].shape[0]
        self.max_size *= self.up

    def rotate_img(self, imgarr, angle):
        image_center = tuple(np.array(imgarr.shape[:2]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(imgarr, rot_mat, imgarr.shape[1::-1], flags=self.interpolation,
                                borderMode=cv2.BORDER_REFLECT_101, borderValue=(0, 0, 0,))
        return result

    def flip_img(self, imgarr, flip):
        imgarr = cv2.flip(imgarr, flip)
        return imgarr

    def adjust_gamma(self, imgarr, gamma):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype(np.uint8)
        return cv2.LUT(imgarr, table)

    def kernel_motion_blur(self, kernel_size, angle):
        '''
        param filter_size: integer, size of square blur filter.
        param angle: integer from range [-90,90]. clockwise angle between horizontal line and target line
            angle=0 means vertical blur
            angle=90 means horizontal blur
            angle>0 means downleft - upright blur
            angle<0 means upleft - downright blur
        returns: filter array of 0s and 1s of size (filter_size x filter_size) with normalization koefficient 1/filter_size
        '''
        if angle > 90:
            angle = 90
        elif angle < -90:
            angle = -90
        kernel_size = int(kernel_size)
        ab_angle = abs(angle)
        start_point = (0, 0)
        end_point = (kernel_size, np.int(kernel_size * np.tan(np.radians(min(ab_angle, 90 - ab_angle)))))
        kernel_motion = np.zeros((kernel_size, kernel_size))
        kernel_motion = cv2.line(kernel_motion, start_point, end_point, 1, 1)
        kernel_motion = kernel_motion / kernel_size
        if angle < -45:
            return np.flip(np.transpose(kernel_motion), 0)
        elif angle >= -45 and angle < 0:
            return np.flip(kernel_motion, 1)
        elif angle >= 0 and angle <= 45:
            return kernel_motion
        else:
            return np.transpose(kernel_motion)

    def motion_blur(self, imgarr):
        h, w, _ = self.dim
        max_dim = h if h >= w else w
        min_ksize = 3 + round(max_dim / 128) if max_dim != 128 else 3  # для 128х128 3 - 5
        max_ksize = 5 + round(max_dim / 128) if max_dim != 128 else 5
        rand_ksize = random.randint(min_ksize, max_ksize)
        rand_angle = random.randint(-90, 90)
        kernel_mb = self.kernel_motion_blur(rand_ksize, rand_angle)
        imgarr = cv2.filter2D(imgarr, -1, kernel_mb)

        blur_ksize = round(rand_ksize // 3) if rand_ksize != 2 else 1  # для 128х128 1 - 2
        imgarr = cv2.blur(imgarr, (blur_ksize, blur_ksize), cv2.BORDER_DEFAULT)
        return imgarr

    def blur(self, imgarr):
        h, w, _ = self.dim
        max_dim = h if h >= w else w
        min_ksize = 3 + round(max_dim / 128) if max_dim != 128 else 3  # для 128х128 3 - 5
        max_ksize = 5 + round(max_dim / 128) if max_dim != 128 else 5
        rand_ksize = random.randint(min_ksize, max_ksize)
        blur_ksize = round(rand_ksize // 3) if rand_ksize != 2 else 1  # для 128х128 1 - 2
        imgarr = cv2.blur(imgarr, (blur_ksize, blur_ksize), cv2.BORDER_DEFAULT)
        return imgarr

    def resize_whithout_distortion(self, img):
        (h, w) = img.shape[:2]
        if h > w:
            nip_off_pecent = int(h * 0.12)
            cut_img = img[nip_off_pecent:h - nip_off_pecent, :]
            (h_c, w_c) = cut_img.shape[:2]
            img = cut_img[:h // 2, w // 2 - w // 4:w_c // 2 + w // 4]
        elif h < w:
            nip_off_pecent = int(w * 0.12)
            cut_img = img[:, nip_off_pecent:w - nip_off_pecent]
            (h_c, w_c) = cut_img.shape[:2]
            img = cut_img[:, w_c // 2 - ((h ** 2) // w) // 2:w_c // 2 + ((h ** 2) // w) // 2]
        else:
            nip_off_pecent = int(w * 0.12)
            cut_img = img[nip_off_pecent:h - nip_off_pecent, :]
            img = cut_img[:, w // 2 - ((h ** 2) // w) // 4:w // 2 + ((h ** 2) // w) // 4]
        return img

    def img_to_array(self, path_to_img):
        (h, w) = self.dim[:2]
        try:
            img = cv2.imread(path_to_img)
            assert np.any(img), f'Ошибка открытия {path_to_img}'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except AssertionError:
            try:
                img = io.imread(path_to_img)
                assert img.shape[-1] in [1, 2, 3, 4], ('Невозможно конвертировать в RGB')
            except:
                try:
                    img = Image.open(path_to_img)
                    img = img.convert('RGB')
                    img = np.array(img, np.uint8)
                except:
                    img = np.zeros((self.dim))
        if img.shape[-1] == 4:
            img = img[:, :, :-1]
        if self.save_distortion_in_preproc:
            if np.random.choice(2, 1, p=[0.8, 0.2])[0]:
                img = self.resize_whithout_distortion(img)
        if self.resize:
            img = cv2.resize(img, (w, h), interpolation=self.interpolation)

        if self.name_sample == 'TRAIN':
            go_random = np.random.choice(5, 5, p=[0.725, 0.25, 0.015, 0.005, 0.005])
            go_random = np.unique(go_random)
            if 0 in go_random:
                try:
                    img = self.adjust_gamma(img, random.uniform(0.5, 1.5))
                except:
                    pass
            if 1 in go_random:
                try:
                    angle = np.random.normal(0, 90, 1)
                    angle = angle[(angle >= -180) & (angle <= 180)]
                    angle = int(np.round(angle, 0))
                    img = self.rotate_img(img, angle)
                except:
                    pass
            if 2 in go_random:
                try:
                    img = self.motion_blur(img)
                except:
                    pass
            if 3 in go_random:
                try:
                    img = self.flip_img(img, 1)
                except:
                    pass
            if 4 in go_random:
                try:
                    img = self.flip_img(img, -1)
                except:
                    pass

        img = img.astype(np.float32)
        img /= 128.
        img -= 1.
        return img

    def get_smooth_label(self, categorical_labels, smooth_factor=0.2):
        new_y = []
        for y_ in categorical_labels:
            y = y_ * (1 - smooth_factor)
            y += (smooth_factor / categorical_labels.shape[-1])
            new_y.append(y)
        new_y = np.array(new_y, dtype=np.float16)
        return new_y


class ImageGenerator_yield(MainGen):
    def __init__(self, path_to_dir, name_sample, dim, batch, trainValidSize=0.2, trainTestSize=0.05, shuffle=True,
                 random_state=41, to_categorical=False, smooth=False, resize=False,
                 interpolation=cv2.INTER_CUBIC, y_batch_dtype=np.uint8, up=1):
        super().__init__(path_to_dir, name_sample, dim, batch, trainValidSize, trainTestSize, shuffle,
                         random_state, to_categorical, smooth, resize, interpolation, y_batch_dtype, up)

    def generator(self):
        bath_on_class = self.batch // self.num_classes
        used_inxs = {key: [] for key in [inx for inx in range(self.num_classes)]}
        while True:
            inxs = []
            for class_ in self.db_to_gen['CLASS'].unique():
                if len(used_inxs[class_]) > 0:
                    mask_class = self.db_to_gen['CLASS'] == class_
                    mask_class[used_inxs[class_]] = False
                    mask_class = np.where(mask_class)[0]
                    if mask_class.shape[0] - bath_on_class < bath_on_class:
                        used_inxs[class_] = []
                else:
                    mask_class = self.db_to_gen['CLASS'] == class_
                    mask_class = np.where(mask_class)[0]
                rand_inxs = np.random.choice(mask_class, bath_on_class)
                inxs.extend(rand_inxs)
                used_inxs[class_] += list(rand_inxs)
            inxs = np.array(inxs, dtype='int64')
            np.random.shuffle(inxs)

            x_batch = np.zeros((inxs.size, *self.dim), dtype=np.float32)
            for index, inx in enumerate(inxs):
                x_batch[index] += self.img_to_array(self.db_to_gen['PATH'][inx])
            y_batch = [self.db_to_gen['CLASS'][inx] for inx in inxs]
            y_batch = np.array(y_batch, dtype=self.y_batch_dtype)
            if self.to_categorical:
                y_batch = to_categorical(y_batch, self.num_classes, dtype=self.y_batch_dtype)
            if self.smooth:
                y_batch = self.get_smooth_label(y_batch)
            yield x_batch, y_batch


class ImageGenerator(MainGen, Sequence):
    def __init__(self, path_to_dir, name_sample, dim, batch, trainValidSize=0.2, trainTestSize=0.05, shuffle=True,
                 random_state=41, to_categorical=False, smooth=False, resize=False,
                 interpolation=cv2.INTER_CUBIC, y_batch_dtype=np.uint8, up=1, column_y_labels='CLASS',
                 save_distortion_in_preproc=False):
        super().__init__(path_to_dir, name_sample, dim, batch, trainValidSize, trainTestSize, shuffle, random_state,
                         to_categorical, smooth, resize, interpolation, y_batch_dtype, up, column_y_labels,
                         save_distortion_in_preproc)
        self.bath_on_class = self.batch // self.num_classes
        self.used_inxs = {key: [] for key in [inx for inx in self.db_to_gen.CLASS.unique()]}

    def __len__(self):
        return int(np.floor(self.max_size / self.batch))

    def __getitem__(self, index):
        X, y = self.__datagen()
        return X, y

    def __datagen(self):
        inxs = []
        for class_ in self.db_to_gen['CLASS'].unique():
            if len(self.used_inxs[class_]) > 0:
                mask_class = self.db_to_gen['CLASS'] == class_  # -> bool
                mask_class[self.used_inxs[class_]] = False
                mask_class = np.where(mask_class)[0]
                if mask_class.shape[0] - self.bath_on_class < self.bath_on_class:
                    self.used_inxs[class_] = []
            else:
                mask_class = self.db_to_gen['CLASS'] == class_
                mask_class = np.where(mask_class)[0]
            rand_inxs = np.random.choice(mask_class, self.bath_on_class)#, replace=False)
            inxs.extend(rand_inxs)
            self.used_inxs[class_] += list(rand_inxs)
        inxs = np.array(inxs, dtype='int64')
        np.random.shuffle(inxs)

        x_batch = np.zeros((inxs.size, *self.dim), dtype=np.float32)
        for index, inx in enumerate(inxs):
            x_batch[index] += self.img_to_array(self.db_to_gen['PATH'][inx])
        y_batch = [self.db_to_gen[self.column_y_labels][inx] for inx in inxs]
        y_batch = np.array(y_batch, dtype=self.y_batch_dtype)
        if self.to_categorical:
            y_batch = to_categorical(y_batch, self.num_classes, dtype=self.y_batch_dtype)
        if self.smooth:
            y_batch = self.get_smooth_label(y_batch)
        return x_batch, y_batch
