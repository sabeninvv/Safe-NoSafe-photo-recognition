from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
from functools import lru_cache as cache
from PIL import Image
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.compat.v2.nn import relu6


def reCreate_model(path_h5, path_json=None, weights=True):
    if weights:
        model = load_model(path_h5,
                           custom_objects={'relu6': relu6}
                           )
    else:
        with open(path_json, 'r') as json_file:
            model = json_file.read()
            model = model_from_json(model)
        model.load_weights(path_h5)
    return model


@cache(maxsize=None)
def crop2img2arr(path, in_shape=(128, 128), crop=False):
    img = Image.open(path, 'r')
    if crop:
        if img.size[0] >= img.size[1]:
            img = img.crop((0, 0, img.size[0], img.size[0]))
        else:
            img = img.crop((0, 0, img.size[1], img.size[1]))
    img = img.resize(in_shape)
    img = img.convert('RGB')
    imgarr = np.array(img, dtype='float64')
    imgarr = imgarr - imgarr.mean()
    imgarr = imgarr / max(imgarr.max(), abs(imgarr.min()))
    return imgarr


def getConfMatrix(model, X_Test, y_Test, to_categorical=False):
    if to_categorical:
        num_pics_test = np.unique(y_Test, return_counts=True, axis=-2)[1]
    else:
        num_pics_test = np.unique(y_Test, return_counts=True)[1]
    data = {'y_Actual': y_Test,
            'y_Predicted': [p.argmax() for p in model.predict(X_Test)]
            }
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'],
                                   df['y_Predicted'],
                                   rownames=['Actual'],
                                   colnames=['Predicted'],
                                   margins=True)
    print('====================================')
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
    print('====================================')
    for row, countInClass in enumerate(num_pics_test):
        confusion_matrix.loc[row] = (confusion_matrix.loc[row] * 100) / countInClass
        confusion_matrix.loc[row] = np.around(confusion_matrix.loc[row], 3)
    return confusion_matrix


def compare_images(model_for_vis, path, t):
    plt.figure(figsize=(14, 6))
    img1 = Image.open(path).resize((200, 200))

    plt.subplot(121)
    plt.imshow(img1)
    plt.axis("off")

    plt.subplot(122)

    arr = crop2img2arr(path)
    img2pred = np.expand_dims(arr, axis=0)
    activation = model_for_vis.predict(img2pred)

    images_per_row = 1
    n_filters = activation.shape[-1]
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
            display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = (display_grid[
                                                                                        col * size: (col + 1) * size,
                                                                                        row * size: (row + 1) * size] + channel_image) / 2
    scale = 0.02
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='bwr')
    plt.axis("off")
    plt.title("Predict: " + str(t), fontsize=18)


def compare_images2(model_for_vis1, model_for_vis2, path, t):
    arr = crop2img2arr(path)
    img2pred = np.expand_dims(arr, axis=0)

    plt.figure(figsize=(21, 6))
    img1 = Image.open(path).resize((200, 200))

    plt.subplot(131)
    plt.imshow(img1)
    plt.axis("off")

    plt.subplot(132)
    activation = model_for_vis1.predict(img2pred)

    images_per_row = 1  # 16
    n_filters = activation.shape[-1]
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
            display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = (display_grid[
                                                                                        col * size: (col + 1) * size,
                                                                                        row * size: (row + 1) * size] + channel_image) / 2

    scale = 0.02
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='bwr')
    plt.axis("off")
    plt.title("Predict: " + str(t), fontsize=18)

    plt.subplot(133)
    activation = model_for_vis2.predict(img2pred)

    images_per_row = 1
    n_filters = activation.shape[-1]
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
            display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = (display_grid[
                                                                                        col * size: (col + 1) * size,
                                                                                        row * size: (row + 1) * size] + channel_image) / 2

    scale = 0.02
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='bwr')
    plt.axis("off")