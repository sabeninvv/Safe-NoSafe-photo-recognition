# -*- coding: utf-8 -*-

import os
from PIL import Image
import cv2
from skimage import io
import sys
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from subprocess import Popen
from time import time


class ImgResizeSave():
    def __init__(self, dir_to_open, new_dir_to_save, dim=None, cut=False,
                 interpolation=cv2.INTER_CUBIC, crop_length=None, number_of_multiproces=0):
        self.dim = dim
        self.interpolation = interpolation
        self.number_of_multiproces = number_of_multiproces
        self.dir_to_open = dir_to_open
        self.new_dir_to_save = new_dir_to_save
        self.cut = cut
        self.crop_length = crop_length


    def rotate_img(self, imgarr, angle):
        image_center = tuple(np.array(imgarr.shape[:2]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(imgarr, rot_mat, imgarr.shape[1::-1], flags=self.interpolation,
                                borderMode=cv2.BORDER_REFLECT_101, borderValue=(0, 0, 0,))
        return result

    def open_img(self, path_to_img):
        flagBGR = False
        try:
            img = cv2.imread(path_to_img)
            assert np.any(img), f'Ошибка открытия {path_to_img}'
        except AssertionError:
            try:
                flagBGR = True
                img = io.imread(path_to_img)
                assert img.shape[-1] in [1, 2, 3, 4], ('Невозможно конвертировать в RGB')
            except:
                try:
                    img = Image.open(path_to_img)
                    img = img.convert('RGB')
                    img = np.array(img, np.uint8)
                except:
                    img = None
        return img, flagBGR

    def resize_img(self, img):
        # initialize the dimensions of the image to be resized and
        (h, w) = img.shape[:2]
        if self.crop_length > h or self.crop_length > w:
            return img
        max_length = h if h > w else w
        # calculate the ratio of the height and construct the dimensions
        ratio = self.crop_length / float(max_length)
        dim = (int(w * ratio), self.crop_length) if h > w else (self.crop_length, int(h * ratio))
        resized_img = cv2.resize(img, dim, interpolation=self.interpolation)
        return resized_img

    def save_img(self, img, rand_name, path_to_save, inx=""):
        if self.dim:
            w_resize, h_resize, _ = self.dim
            img = cv2.resize(img, (w_resize, h_resize), interpolation=self.interpolation)
        if self.crop_length:
            img = self.resize_img(img)
        name = rand_name + f'{inx}.jpg'
        path_to_new_img = os.path.join(path_to_save, name)
        cv2.imwrite(path_to_new_img, img, [cv2.IMWRITE_JPEG_QUALITY, 100])

    def img_resize_and_save(self, paths_to_imgs):
        path_to_img, path_to_save = paths_to_imgs
        path_to_save = os.path.dirname(path_to_save)
        img, flagBGR = self.open_img(path_to_img)
        orient_H = False
        if np.any(img):
            if img.shape[-1] == 4:
                img = img[:, :, :-1]
            if flagBGR:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rand_name = str(int(time()))
            rand_name += ''.join(np.random.choice('h f k s l y'.split(' '), 5))
            h, w = (img.shape[0], img.shape[1])
            if self.cut and h != w:
                if h < w:
                    img = np.rot90(img)
                    h, w = (w, h)
                    orient_H = True
                left, right = (0, w)
                up_down = [(0, w), (h - w, h), (h // 4, h // 4 + w) if  h // 4 + w < h else (h - w, h)]
                for inx, i in enumerate(up_down):
                    img_crop = img[i[0]:i[1], left:right]
                    if orient_H:
                        img_crop = self.rotate_img(img_crop, -90)
                    self.save_img(img=img_crop, rand_name=rand_name, path_to_save=path_to_save, inx=inx)
            else:
                self.save_img(img=img, rand_name=rand_name, path_to_save=path_to_save)

    def imgs_copy(self):
        if self.number_of_multiproces:
            args = []
            args.append(str(self.dir_to_open))
            args.append(str(self.new_dir_to_save))
            if self.dim:
                args.append(f'{self.dim[0]},{self.dim[1]},{self.dim[2]}')
            else:
                args.append('')
            if self.crop_length:
                args.append(str(self.crop_length))
            else:
                args.append('')
            if self.cut:
                args.append(str(self.cut))
            else:
                args.append('')
            args.append(str(self.number_of_multiproces))
            print(args)
            with Popen([sys.executable, str(__file__), *args]) as process:
                _ = process.communicate()
        else:
            for dir_name in os.listdir(self.dir_to_open):
                child_dir = os.path.join(self.dir_to_open, dir_name)
                if os.path.isdir(child_dir) and dir_name.isdigit():
                    new_child_dir = os.path.join(self.new_dir_to_save, dir_name)
                    if not os.path.exists(new_child_dir):
                        os.makedirs(new_child_dir)
                    paths_to_imgs = [[os.path.join(child_dir, img), os.path.join(new_child_dir, img)] for img in
                                     os.listdir(child_dir)]
                    for paths in tqdm(paths_to_imgs):
                        self.img_resize_and_save(paths)


class new_class(ImgResizeSave):
    def imgs_copy(self):
        for dir_name in os.listdir(self.dir_to_open):
            child_dir = os.path.join(self.dir_to_open, dir_name)
            if os.path.isdir(child_dir) and dir_name.isdigit():
                new_child_dir = os.path.join(self.new_dir_to_save, dir_name)
                if not os.path.exists(new_child_dir):
                    os.makedirs(new_child_dir)
                paths_to_imgs = [[os.path.join(child_dir, img), os.path.join(new_child_dir, img)] for img in
                                 os.listdir(child_dir)]
                with Pool(self.number_of_multiproces) as pool:
                    pool.map(self.img_resize_and_save, paths_to_imgs)


def main(*args):
    if args:
        dir_to_open, new_dir_to_save, dim, crop_length, cut, number_of_multiproces = args[0]
        dim = tuple(np.array(dim.split(','), dtype=np.uint8)) if dim else None
        crop_length = int(crop_length) if crop_length else None
        cut = bool(cut) if cut else None
        number_of_multiproces = int(number_of_multiproces)
        class_obj = new_class(dir_to_open=dir_to_open, new_dir_to_save=new_dir_to_save, cut=cut,
                              dim=dim, crop_length=crop_length, number_of_multiproces=number_of_multiproces)
        class_obj.imgs_copy()


if __name__ == '__main__':
    main(sys.argv[1:])
