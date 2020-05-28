import numpy as np
import cv2
import av
from _tools import rotate_img
from numba import njit
import av.datasets
import logging
logging.getLogger('libav').setLevel(logging.ERROR)


@njit
def get_denominator(stream_frames, frames_count, fps):
    '''
    Расчёт коэфициента для расчёта шага распределения.
    :param stream_frames: int
    :param frames_count: int
    :param fps: int
    :return: float
    '''
    arg = np.ceil(stream_frames / frames_count)
    denominator = np.ceil(fps / arg)
    return denominator + .3

def blur_detection(arr):
    '''
    Blur detection with Laplacian operator.
    :param arr: np.array.
    :return: np.float16.
    '''
    lapl = cv2.Laplacian(arr, cv2.CV_16S, ksize=1).var()
    return lapl.astype(np.float16)


def parse(arr, element, step):
    '''
    Проверка пересечения нового элемента с элементами в массиве при условии заданного шага.
    Все индексы не должны  пересекаться согласно заданного шага.
    :param arr: np.array. Массив индексов видеофрейма.
    :param element: int. Новый индекс.
    :param step: int. Шаг.
    :return: bool.
    '''
    new_arr = np.arange(0 if element - step < 0 else element - step, element + step + 1)
    for elem in new_arr:
        if elem in arr:
            return False
    return True


def frame_to_arr(frame, dim, normalize, interpolation, reshape, rotate):
    '''
    Конвертация видеофрейма в np.array
    :param frame: object. av.video.frame.VideoFrame.
    :param dim: tuple. Размерность изображения.
    :param normalize: bool. Нормализация.
    :param interpolation: object. Необходимая интерполяция.
    :param reshape: bool. Изменение размерности. Необходимо для np.append.
    :param rotate: float. Угол поворота.
    :return: np.array
    '''
    img = frame.reformat(format="yuv420p").reformat(format="rgb24").to_ndarray()
    if np.any(dim):
        h, w, _ = dim
        img = cv2.resize(img, (h, w), interpolation=interpolation)
    if normalize:
        img = img.astype(np.float32)
        img /= 128.
        img -= 1.
    if rotate:
        img = rotate_img(imgarr=img, angle=rotate, interpolation=interpolation)
    if reshape:
        img = img.reshape((1, *dim))
    return img


def get_inxs(path, dim, frames_count, denominator, interpolation=cv2.INTER_AREA):
    '''
    Проход по всему видеофайлу.
    Ранжирование индексов фреймов относительно Laplacian и распределение согласно полученного шага.
    :param path: str. Путь к видеофайлу.
    :param dim: tuple. Размерность изображения.
    :param frames_count: int. Сколько фреймов изъять из видео.
    :param denominator: float. Коэфициент для расчёта шага распределения.
    :param interpolation: object. Необходимая интерполяция.
    :return: np.array
    '''
    with av.open(path) as container:
        stream = container.streams.video[0]
        fps = np.ceil(stream.base_rate.numerator / stream.base_rate.denominator)
        if not denominator:
            denominator = get_denominator(stream_frames=stream.frames, frames_count=frames_count, fps=fps)
        laplacians = np.zeros(0, np.float16)
        for frame in container.decode(stream):
            arr = frame_to_arr(frame=frame, dim=dim, normalize=False, interpolation=interpolation, reshape=False,
                               rotate=None)
            laplacian = blur_detection(arr)
            laplacians = np.append(laplacians, laplacian, axis=None)
        # Сортировка от меньшего
        args_sort = np.argsort(laplacians)
        # Сортировка от большего
        revers_args_sort = np.flip(args_sort, axis=0)
        inxs = np.array([], np.int32)
        for inx in revers_args_sort:
            if parse(arr=inxs, element=inx, step=np.floor(fps / denominator)):
                inxs = np.append(inxs, inx)
            if inxs.size == frames_count:
                break
        return inxs

def video_to_arrays(path, frames_count, dim=None, denominator=None, interpolation=cv2.INTER_CUBIC, normalize=True,
                    rotate=False):
    '''
    Создание np.array на основе ранжирования индексов фреймов относительно Laplacian
    и распределение согласно полученного шага.
    :param path: str. Путь к видеофайлу.
    :param frames_count: int. Сколько фреймов изъять из видео.
    :param dim: tuple. Размерность изображения.
    :param denominator: float. Коэфициент для расчёта шага распределения.
    :param interpolation: object. Необходимая интерполяция.
    :param normalize: bool. Нормализация.
    :param rotate: float. Автоматический поворот фрейма.
    :return: np.array
    '''
    assert dim, 'Введите размерность dim.'
    assert frames_count, 'Введите количество получаемых кадров frames_count.'
    inxs = get_inxs(path=path, dim=dim, frames_count=frames_count, denominator=denominator)
    arrs = np.zeros((0, *dim), np.float32)

    with av.open(av.datasets.curated(path)) as container:
        container.streams.video[0].thread_type = 'FRAME'
        if rotate and 'rotate' in container.streams.video[0].metadata.keys():
            angle = -int(container.streams.video[0].metadata['rotate'])
        else:
            angle = None

        for packet in container.demux():
            if isinstance(packet.stream, av.video.stream.VideoStream):
                for frame in packet.decode():
                    if frame.index in inxs:
                        arr = frame_to_arr(frame=frame, dim=dim, normalize=normalize, interpolation=interpolation, reshape=True,
                                           rotate=angle)
                        arrs = np.concatenate((arrs, arr), axis=0)
                    if arrs.shape[0] == frames_count:
                        break
        return arrs
