from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from _tools import getConfMatrix, crop2img2arr
from _video import video_to_arrays
from abc import ABC, abstractmethod


class AbstractClass(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def make_db(self) -> None:
        pass

    @abstractmethod
    def get_confmatrix(self) -> None:
        pass

    @abstractmethod
    def write_data(self) -> None:
        pass


class MainClass(AbstractClass):
    def __init__(self, model, path_to_dir, batch, dim, interpolation, normalize, filename):
        super().__init__()
        self.model = model
        self.batch = batch
        self.path_to_dir = path_to_dir
        self.interpolation = interpolation
        self.normalize = normalize
        self.w, self.h, self.c = dim
        self.filename = filename
        self.probability = np.zeros((0, 0), np.float16)
        self.y_true = np.array([], dtype=np.uint8)

    def make_db(self):
        paths_to_files = []
        labels = []
        for dir_name in os.listdir(self.path_to_dir):
            child_dir = os.path.join(self.path_to_dir, dir_name)
            if os.path.isdir(child_dir) and dir_name.isdigit():
                for img_name in os.listdir(child_dir):
                    path_to_file = os.path.join(child_dir, img_name)
                    if os.path.isfile(path_to_file):
                        paths_to_files.append(path_to_file)
                        labels.append(int(dir_name))
        self.db = pd.DataFrame(data={'PATH': paths_to_files,
                                     'CLASS': labels})
        self.num_classes = self.db['CLASS'].nunique()
        if self.num_classes != self.model.output.shape.as_list()[-1]:
            self.num_classes = self.model.output.shape.as_list()[-1]
        self.probability = np.zeros((0, self.num_classes), np.float16)

    def get_confmatrix(self, y_pred=None, prefix='', ):
        if not np.any(y_pred):
            y_pred = np.zeros((self.probability.shape[0]), np.uint16)
            for inx, p in enumerate(self.probability):
                y_pred[inx] = p.argmax()
        confmatrix = getConfMatrix(model=self.model, y_true=self.y_true, y_pred=y_pred, X_test=None)
        self.write_data(confmatrix, prefix)

    def write_data(self, confmatrix, prefix):
        with open(self.filename, 'a+', encoding='utf8') as f:
            f.write(f'#### ConfMatrix from {prefix} in %\n\n')
            f.write(f'% |')
            f.write(''.join([f'   {i}  |' for i in confmatrix.keys()]))
            f.write('\n---|')
            f.write(''.join(['---|' for _ in confmatrix.keys()]))
            f.write('\n')
            for inx, i in zip(confmatrix[0].index, confmatrix.values):
                f.write(f'{inx} | ')
                f.write(' | '.join(i.astype('str')))
                f.write('\n')
            f.write('\n\n')

class Pred2imgs(MainClass):
    def __init__(self, model, path_to_dir, dim, batch=50,
                 interpolation=cv2.INTER_CUBIC, normalize=True, filename='temp.md',
                 prefix='FOTO'):
        super().__init__(model, path_to_dir, batch, dim, interpolation, normalize, filename)
        self.prefix = prefix
        self.arrs_to_batch = np.zeros((0, self.w, self.h, self.c), np.float32)
        self.make_db()

    def predict_from_batch(self):
        probability = self.model.predict(self.arrs_to_batch)
        self.probability = np.concatenate((self.probability, probability), axis=0)
        self.arrs_to_batch = np.zeros((0, self.w, self.h, self.c), np.float32)

    def img_to_arrays(self, path, class_):
        arr_img = crop2img2arr(path_to_img=path, dim=(self.w, self.h, self.c),
                               normalize=True, interpolation=self.interpolation)
        arr_img = arr_img.reshape((1, *arr_img.shape))
        self.arrs_to_batch = np.concatenate((self.arrs_to_batch, arr_img), axis=0)
        self.y_true = np.append(self.y_true, class_)
        if self.arrs_to_batch.shape[0] == self.batch:
            self.predict_from_batch()

    def predict(self):
        print('Start to predict images')
        for row in tqdm(self.db.values):
            self.img_to_arrays(path=row[0], class_=row[1])
        if self.arrs_to_batch.shape[0] != 0:
            self.predict_from_batch()
        self.get_confmatrix(prefix=self.prefix)
        print('Completed\n')


class Pred2vds(MainClass):
    def __init__(self, model, path_to_dir, dim, batch=50,
                 interpolation=cv2.INTER_CUBIC, limit=0.5,
                 normalize=True, frames_count=3, filename='temp.md', prefix='VIDEO',
                 rotate=False):
        super().__init__(model, path_to_dir, batch, dim, interpolation, normalize, filename)
        self.prefix = prefix
        self.frames_count = frames_count
        self.limit = limit
        self.rotate = rotate
        self.frame_counter = 0
        self.arrs_to_batch = np.zeros((0, self.w, self.h, self.c), np.float32)
        self.y_pred = np.array([], dtype=np.uint16)
        self.make_db()

    def fill_dict(self, arrs, videoframes_and_inxs, inx):
        self.frame_counter = arrs.shape[0]
        start = self.arrs_to_batch.shape[0] - self.frame_counter
        finish = self.arrs_to_batch.shape[0]
        videoframes_and_inxs[inx] = np.arange(start, finish)
        return videoframes_and_inxs

    def fill_y_true(self, label):
        tail = np.array([label] * self.frame_counter, np.uint8)
        self.y_true = np.append(self.y_true, tail)

    def check_condition(self, videoframes_and_inxs, condition):
        if condition:
            self.predict_batch(videoframes_and_inxs)
            self.arrs_to_batch = np.zeros((0, self.w, self.h, self.c), np.float32)
            videoframes_and_inxs = dict()
        return videoframes_and_inxs

    def predict_batch(self, videoframes_and_inxs):
        propability_batch = self.model.predict(self.arrs_to_batch)
        for key in videoframes_and_inxs.keys():
            propability_video = propability_batch[videoframes_and_inxs[key]][:, 0]
            if np.any(propability_video > self.limit):
                tail = np.array([0] * propability_video.shape[0], np.uint8)
                self.y_pred = np.append(self.y_pred, tail)
            else:
                tail = np.array([1] * propability_video.shape[0], np.uint8)
                self.y_pred = np.append(self.y_pred, tail)

    def predict(self):
        videoframes_and_inxs = dict()
        print('Start to predict videoframes')
        for inx, row in enumerate(tqdm(self.db.values)):
            arrs = video_to_arrays(path=row[0], dim=(self.w, self.h, self.c), frames_count=self.frames_count, rotate=self.rotate)
            self.arrs_to_batch = np.concatenate((self.arrs_to_batch, arrs), axis=0)
            videoframes_and_inxs = self.fill_dict(arrs=arrs, videoframes_and_inxs=videoframes_and_inxs, inx=inx)
            self.fill_y_true(label=row[1])
            videoframes_and_inxs = self.check_condition(videoframes_and_inxs=videoframes_and_inxs,
                                                        condition=(self.arrs_to_batch.shape[
                                                                       0] + self.frames_count >= self.batch))
        _ = self.check_condition(videoframes_and_inxs=videoframes_and_inxs, condition=(self.arrs_to_batch.shape[0]!=0))
        self.get_confmatrix(prefix=self.prefix, y_pred=self.y_pred)
        print('Completed\n')
