from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compat.v2.nn import relu6
from tensorflow.keras.activations import hard_sigmoid
from tensorflow.keras.layers import DepthwiseConv2D, GlobalAveragePooling2D, Conv2D,  concatenate, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization, Activation, multiply, ZeroPadding2D


def _upDeep(inputs, pointwise_conv, activation=None):
    '''
    Создаёт слой, увеличивающий количество фильтров
    :param inputs: входной тензор
    :param pointwise_conv: новое количество фильтров
    :param activation: функция активации
    :return:
    '''
    x = Conv2D(pointwise_conv,
               (1, 1),
               strides=(1, 1),
               padding='same',
               use_bias=False,
               activation=activation)(inputs)
    return x


def _conv2d(inputs, deep, strides=(1, 1), activation=None, pool_size=None):
    '''
    Обработка свёрточным слоем.
    Функция добавляет: DepthwiseConv2D() -> Activation() -> Dropout()
    Для оптимизации обработки используется свёрточные ядра:
    - 3х1 -> 1x3 => 3x3
    - 3x3 -> 3x3 => 5x5
    - 3x3 -> 3x3 -> 3x3 => 7x7
    :param inputs: входной тензор
    :param deep: количество итераций обработки свёрткой 3х3
    :param strides: шаг свёртки вдоль оХ и oY. strides(2,2) - сожмёт картинку в 2 раза
    :param activation: функция активации
    :param pooling_size: ядро свёртки при strides(2,2)
    :return:
    '''
    if strides == 2:
        if deep == 1:
            x = ZeroPadding2D(((0, pool_size - 2), (0, pool_size - 2)))(inputs)
        else:
            x = ZeroPadding2D(((0, pool_size - 2), (0, pool_size - 2)))(inputs)
        x = DepthwiseConv2D(pool_size,
                            strides=strides,
                            padding='same' if strides == 1 else 'valid',
                            use_bias=False,
                            activation=None)(x)
        x = BatchNormalization(momentum=0.999, epsilon=0.001)(x)
        x = Activation(relu6)(x)
        x = Dropout(0.2)(x)
    else:
        for wave in range(deep):
            if wave >= 1:
                inputs = x
            if strides == 1:
                x = DepthwiseConv2D((3, 1),
                                    strides=strides,
                                    padding='same',
                                    use_bias=False,
                                    activation=None)(inputs)
                x = DepthwiseConv2D((1, 3),
                                    strides=(1, 1),
                                    padding='same',
                                    use_bias=False,
                                    activation=None)(x)
            if wave == deep - 1:
                x = BatchNormalization(momentum=0.999, epsilon=0.001)(x)
            if not activation:
                x = Activation(relu6)(x)
            if wave == deep - 1:
                x = Dropout(0.2)(x)
    return x


def _multyConv2d(inputs, strides=(1, 1), deep1=1, deep2=2):
    '''
    Функция:
    1. Создаёт ансамбль из 2 веток:
    - обработка ядром 3х3
    - обработка ядром 5х5
    2. Объеденяет результаты веток (увеличение количества фильтров)
    3. Схлапывает до глубины входного тензора (inputs.shape[-1]). см. _upDeep
    :param inputs: входной тензор
    :param strides: шаг свёртки вдоль оХ и oY. strides(2,2) - сожмёт картинку в 2 раза
    :param deep1: количество итераций обработки свёрткой 3х3. см. _conv2d
    :param deep2: количество итераций обработки свёрткой 3х3. см. _conv2d
    :return:
    '''
    point = int(inputs.shape[-1])
    if strides == (1, 1):
        x1 = _conv2d(inputs, strides=strides, deep=deep1)
        x2 = _conv2d(inputs, strides=strides, deep=deep2)
    else:
        x1 = _conv2d(inputs, strides=strides, deep=deep1, pool_size=3)
        x2 = _conv2d(inputs, strides=strides, deep=deep2, pool_size=5)
    x = concatenate([x1, x2], axis=-1)
    # project
    x = _upDeep(x, pointwise_conv=point, activation=None)
    return x


def _multyConv2d_res(inputs, residual, strides=(1, 1), deep1=1, deep2=2, momentum=0.99):
    '''
    Функция:
    1. Создаёт ансамбль из 2 веток:
    - обработка ядром 3х3
    - обработка ядром 5х5
    2. Объеденяет результаты веток и residual_block (увеличение количества фильтров)
    3. Схлапывает до глубины входного тензора (inputs.shape[-1]). см. _upDeep
    :param inputs: входной тензор
    :param residual: входной residual блок
    :param strides: шаг свёртки вдоль оХ и oY. strides(2,2) - сожмёт картинку в 2 раза
    :param deep1: количество итераций обработки свёрткой 3х3. см. _conv2d
    :param deep2: количество итераций обработки свёрткой 3х3. см. _conv2d
    :return:
    '''
    point = int(inputs.shape[-1])
    if strides == (1, 1):
        x1 = _conv2d(inputs, strides=strides, deep=deep1)
        x2 = _conv2d(inputs, strides=strides, deep=deep2)
    else:
        x1 = _conv2d(inputs, strides=strides, deep=deep1, pool_size=3)
        x2 = _conv2d(inputs, strides=strides, deep=deep2, pool_size=5)
    # Определяем необходимость "сжатия" входного residual блока
    # iHW - высота и ширина inputs
    # rHW - высота и ширина residual
    # если они равны, то strides_res (результат деления) даст 1,
    # иначе, strides_res получит шаг свёртки для сжатия картинки до размеров inputs (входного тензора)
    iHW = (x1.shape[1], x1.shape[2])
    iHW = np.array(iHW, 'int16')
    rHW = (residual.shape[1], residual.shape[2])
    rHW = np.array(rHW, 'int16')
    strides_res = np.array(rHW / iHW, 'int16')

    if np.all(strides_res == 1):
        x = concatenate([x1, x2, residual], axis=-1)
    else:
        residual = DepthwiseConv2D((1, 1),
                                   strides=strides_res,
                                   padding='same',
                                   use_bias=False,
                                   activation=None)(residual)
        x = concatenate([x1, x2, residual], axis=-1)
    # project
    x = _upDeep(x, pointwise_conv=point, activation=None)
    return x


def _residual(inputs, strides=(1, 1), activation='linear'):
    '''
    Создание и калибровка residual block
    :param inputs: входной тензор
    :param strides: шаг свёртки вдоль оХ и oY. strides(2,2) - сожмёт картинку в 2 раза
    :param activation: функция активации
    :return:
    '''
    # Создание residual block
    point = int(inputs.shape[-1])
    x = DepthwiseConv2D((1, 1),
                        strides=strides,
                        padding='same',
                        use_bias=False,
                        activation=activation)(inputs)
    # Калибровка
    # см. squeeze-and-excitation
    x1 = GlobalAveragePooling2D()(x)
    x1 = Dense(point // 16,
               use_bias=False,
               activation=relu6)(x1)
    x1 = Dense(point,
               use_bias=False,
               activation=hard_sigmoid)(x1)
    x = multiply([x, x1])
    return x


def _get_percent(point, percent):
    '''
    Получает количество фильтров входного тензора.
    Вычисляет целочисленное деление на 8.
    (обоснованность использьваония количества фильтров кратных 8, обусловленно эмперическим путём)
    Коэфициент полученный от целочисленного деления умножается на 8.
    В данной архитектуре не используется.

    :param point: количество фильтров входного тензора
    :param percent: процент от point для создания нового свёрточного слоя
    :return:
    '''
    cvant = (point * percent) // 8.
    if cvant != 0:
        return int(cvant * 8)
    else:
        return 8


def get_block(inputs, residual, pointwise_conv, conv_iter=1, activation_up='linear', res_block=False, strides=(1, 1),
              deep1=1, deep2=2, momentum=0.99):
    '''
    Функция:
    1. Создаёт _upDeep слой. Необходим для увеличения количества фильтров. см. _upDeep
    2. Объеденяет результаты ансамбля свёрток deep1 иdeep2 и residual_block (вариативно)
    3. Создаёт и калибрует новый residual block

    :param inputs: входной тензор
    :param residual: входной residual блок
    :param pointwise_conv:
    :param conv_iter: количество свёрточных слоёв
    :param activation_up: функция активации "расширяющего" слоя. см. _upDeep
    :param res_block: наличие residual блока
    :param strides: шаг свёртки вдоль оХ и oY. strides(2,2) - сожмёт картинку в 2 раза
    :param deep1: количество итераций обработки свёрткой 3х3. см. _conv2d
    :param deep2: количество итераций обработки свёрткой 3х3. см. _conv2d
    :return:
    '''
    x = _upDeep(inputs,
                pointwise_conv=pointwise_conv,
                activation=activation_up)

    for _ in range(conv_iter):
        if res_block:
            x = _multyConv2d_res(x, residual=residual, strides=strides, deep1=deep1, deep2=deep2, momentum=momentum)
        else:
            x = _multyConv2d(x, strides=strides, deep1=deep1, deep2=deep2, momentum=momentum)
    residual = _residual(x)
    return x, residual