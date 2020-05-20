from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import av
from _tools import getConfMatrix, crop2img2arr
from abc import ABC, abstractmethod


class AbstractClass(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def make_db(self) -> None:
        pass

    @abstractmethod
    def get_confmatrix_and_quantiles(self) -> None:
        pass

    @abstractmethod
    def write_data(self) -> None:
        pass


class MainClass(AbstractClass):
    def __init__(self, model, path_to_dir, quantiles, batch, dim, interpolation, resize, normalize, filename):
        super().__init__()
        self.model = model
        self.batch = batch
        self.path_to_dir = path_to_dir
        self.quantiles = quantiles
        self.interpolation = interpolation
        self.resize = resize
        self.normalize = normalize
        self.w, self.h, self.c = dim
        self.filename = filename
        self.probability = np.zeros((0, 0), np.float16)
        self.y_true = np.array([], dtype=np.uint16)

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
        self.probability = np.zeros((0, self.num_classes), np.float16)

    def get_confmatrix_and_quantiles(self, prefix=''):
        quantiles = {}
        self.y_pred = np.zeros((self.probability.shape[0]), np.uint16)
        for inx, p in enumerate(self.probability):
            self.y_pred[inx] = p.argmax()
        for key in range(self.num_classes):
            temp = self.probability[np.where(self.y_pred == key)[0]][:, key]
            quantiles[key] = np.quantile(temp, self.quantiles) if np.any(temp) else np.zeros(0, np.uint8)

        confmatrix = getConfMatrix(model=self.model, y_true=self.y_true, y_pred=self.y_pred, X_test=None)
        self.write_data(confmatrix, quantiles, prefix)

    def write_data(self, confmatrix, quantiles, prefix):
        with open(self.filename, 'a+', encoding='utf8') as f:
            f.write(f'#### ConfMatrix from {prefix} in %\n\n')
            f.write(f'% |')
            f.write(''.join([f'   {i}  |' for i in confmatrix.keys()[:-1]]))
            f.write('\n---|')
            f.write(''.join(['---|' for i in confmatrix.keys()[:-1]]))
            f.write('\n')
            for inx, i in enumerate(confmatrix.values[:-1, :-1]):
                f.write(f'{inx} | ')
                f.write(' | '.join(i.astype('str')))
                f.write('\n')
            f.write('\n\n')

        with open(self.filename, 'a+', encoding='utf8') as f:
            f.write(f'#### Quantiles from {prefix}\n\n')
            f.write(f'% |')
            f.write(''.join([f'   {int(i * 100)}  |' for i in self.quantiles]))
            f.write('\n---|')
            f.write(''.join(['---|' for i in self.quantiles]))
            f.write('\n')
            for key in quantiles:
                f.write(f'{key} | ')
                f.write(' | '.join(np.around(quantiles[key], 2).astype('str')))
                f.write('\n')
            f.write('\n\n')


class Pred2imgs(MainClass):
    def __init__(self, model, path_to_dir, dim, quantiles=(.55, .65, .75, .85, .95), batch=100,
                 interpolation=cv2.INTER_CUBIC, resize=True, normalize=True, filename='temp.md',
                 prefix='FOTO'):
        super().__init__(model, path_to_dir, quantiles, batch, dim, interpolation, resize, normalize, filename)
        self.prefix = prefix
        self.make_db()

    def fill_y_true(self, shape, class_):
        y_true = np.zeros(shape, dtype=np.int32)
        y_true.fill(class_)
        self.y_true = np.concatenate((self.y_true, y_true))

    def predict_fromarr(self, arrs_to_pred, class_):
        arrs = arrs_to_pred if self.batch != 1 else np.expand_dims(arrs_to_pred, axis=0)
        probability = self.model.predict(arrs)
        self.probability = np.concatenate((self.probability, probability), axis=0)
        self.probability = np.round(self.probability, 2)
        self.fill_y_true(arrs.shape[0], class_)

    def predict(self):
        print('Start to predict images')
        for class_ in range(self.num_classes):
            val = self.db['PATH'][self.db['CLASS'] == class_].values
            gen = self.gen2pred(val)
            with tqdm(total=len(val) // self.batch) as pbar:
                for arrs_to_pred in gen:
                    self.predict_fromarr(arrs_to_pred, class_)
                    pbar.update()
        self.get_confmatrix_and_quantiles(prefix=self.prefix)
        print('Completed\n')

    def fill_zeros(self, paths):
        arrs_to_pred = np.zeros((len(paths), self.w, self.h, self.c), np.float32)
        for inx, path in enumerate(paths):
            arrs_to_pred[inx] = crop2img2arr(path_to_img=path, dim=(self.w, self.h, self.c), resize=True,
                                             normalize=True, interpolation=self.interpolation)
        return arrs_to_pred

    def gen2pred(self, val):
        start_inx = 0
        end_inx = self.batch
        while len(val) != end_inx:
            if len(val) >= end_inx:
                paths = val[start_inx:end_inx]
                arrs_to_pred = self.fill_zeros(paths)
                (start_inx, end_inx) = (start_inx + self.batch, end_inx + self.batch)
            else:
                end_inx = len(val)
                paths = val[start_inx:end_inx]
                arrs_to_pred = self.fill_zeros(paths)
            yield arrs_to_pred


class Pred2vds(MainClass):
    def __init__(self, model, path_to_dir, dim, quantiles=(.55, .65, .75, .85, .95), batch=100,
                 interpolation=cv2.INTER_CUBIC, resize=True,
                 normalize=True, only_key_frames=True, frames_count=3, filename='temp.md', prefix='VIDEO'):
        super().__init__(model, path_to_dir, quantiles, batch, dim, interpolation, resize, normalize, filename)
        self.prefix = prefix
        self.only_key_frames = only_key_frames
        self.frames_count = frames_count
        self.arrs_to_batch = np.zeros((0, self.w, self.h, self.c), np.float32)
        self.make_db()

    def frame_to_arr(self, frame, class_):
        img = frame.to_image().convert('RGB')
        img = np.array(img, dtype=np.uint8)
        if self.resize:
            img = cv2.resize(img, (self.w, self.h), interpolation=self.interpolation)
        if self.normalize:
            img = img.astype(np.float32)
            img /= 128.
            img -= 1.
        img = img.reshape((1, *img.shape))
        self.arrs_to_batch = np.concatenate((self.arrs_to_batch, img), axis=0)
        self.y_true = np.append(self.y_true, class_)
        if self.arrs_to_batch.shape[0] == self.batch:
            self.predict_from_batch()

    def video_to_arrays(self, path, class_):
        with av.open(path) as container:
            counts = 0
            stream = container.streams.video[0]  # 0 - видео, 1,2 - аудио и субтитры
            stream.codec_context.skip_frame = 'NONKEY' if self.only_key_frames else 'DEFAULT'
            # В stream.frames последний кадр всегда пустой
            self.frames_count = stream.frames - 1 if self.frames_count > stream.frames else self.frames_count
            rand_frames = np.arange(stream.frames) if self.only_key_frames else np.random.choice(stream.frames - 1,
                                                                                                 self.frames_count,
                                                                                                 replace=False)
            for frame in container.decode(stream):
                if frame.index in rand_frames:
                    counts += 1
                    self.frame_to_arr(frame, class_)
                    if counts == self.frames_count:
                        break

    def predict_from_batch(self):
        probability = self.model.predict(self.arrs_to_batch)
        if probability.shape[-1] > self.probability.shape[-1]:
            self.probability = np.zeros((0, probability.shape[-1]), np.float16)
        self.probability = np.concatenate((self.probability, probability), axis=0)
        self.arrs_to_batch = np.zeros((0, self.w, self.h, self.c), np.float32)

    def predict(self):
        print('Start to predict videoframes')
        for row in tqdm(self.db.values):
            self.video_to_arrays(path=row[0], class_=row[1])
        if self.arrs_to_batch.shape[0] != 0:
            self.predict_from_batch()
        self.get_confmatrix_and_quantiles(prefix=self.prefix)
        print('Completed\n')
