import os
import datetime
from time import time
from tensorflow.keras.models import model_from_json
from tensorflow.keras.backend import clear_session

try:
    from tensorflow import reset_default_graph
    from tensorflow.keras.backend import get_session
    from tensorflow.saved_model import simple_save
    from tensorflow import __version__

    print(f'Вы используете TF весии {__version__}')
    print(f'Преобразование .h5 в .pb возможно')
except:
    from tensorflow import __version__

    print(f'Вы используете TF весии {__version__}')
    print(f'Преобразование .h5 в .pb работает в тестовом режиме. Стабильное преобразование возможно на версии TF 1.15')


def get_time_stamp():
    '''
    Создаёт временной штамп.
    time() - исчесление в секундах c January 1, 1970, 00:00:00
    date_stamp.tm_year, date_stamp.tm_mon, date_stamp.tm_mday - год, месяцб день
    :return: string
    '''
    date_stamp = datetime.datetime.now().timetuple()
    time_stamp = int(time())
    time_stamp = f'{date_stamp.tm_year}_{date_stamp.tm_mon}_{date_stamp.tm_mday}_{time_stamp}'
    return time_stamp


def get_version_num(path_model_dir: str):
    '''
    Создаёт новую директорию с последней версией модели
    :param path_model_dir: string
    :return: string
    '''
    path2model_name = os.path.normpath(path_model_dir)
    dirs = os.listdir(path2model_name)
    dirs = [int(i) for i in dirs if i.isdigit()]
    if dirs:
        new_dir = max(dirs) + 1
        new_dir = os.path.join(path2model_name, str(new_dir))
    else:
        new_dir = os.path.join(path2model_name, str(1))
    return new_dir


def make_paths(path_model_dir: str):
    '''
    Создаёт путь для хранения всех каталогов промежуточного уровня и хранения конечного каталога
    path2model_name/pb
    path2model_name/h5
    path2model_name/json
    path2model_name/config
    :param path_model_dir: string. Абсолютный путь
    :return: list
    '''
    paths2dirs = []
    path_model_dir = get_version_num(path_model_dir)
    for dirname in ['h5', 'json', 'config', 'pb']:
        dir_patch = os.path.join(path_model_dir, dirname)
        paths2dirs.append(dir_patch)
    return paths2dirs


def create_dirs(path: str):
    '''
    Создаёт все каталоги промежуточного уровня для хранения конечного каталога
    :param path: string
    '''
    if os.path.isdir(path):
        print(f'Дирректория {path} уже существует')
    else:
        try:
            os.makedirs(path)
        except OSError:
            print(f'Создание директории {path} провалено')
        else:
            print(f'Успешно созданна директория {path}')


def make_dirs(path_model_dir: str):
    '''
    Создаёт все каталоги промежуточного уровня для хранения конечного каталога
    :param path_model_dir: string. Абсолютный путь до модели
    :return: list
    '''
    paths = make_paths(path_model_dir)
    for path in paths:
        create_dirs(path)
    return paths


def save_structure(model: object, path_json: str, name_json='structure.json', txt_structure=None):
    '''
    Сохраняет архитектуру модели.
    :param model: obj. Keras модель
    :param path_json: string
    :param txt_structure: string
    '''
    assert txt_structure, ('Не заполнен txt_structure')
    filename = os.path.dirname(path_json)
    filename = os.path.join(filename, 'structure.txt')
    with open(filename, 'w') as txt_file:
        txt_structure = f'{get_time_stamp()}\n {txt_structure}'
        txt_file.write(txt_structure)
    json_structure = model.to_json()
    filename = os.path.join(path_json, name_json)
    with open(filename, 'w') as json_file:
        json_file.write(json_structure)


def load_structure(path: str, filename='structure.json', **kwargs):
    '''
    Восстанавливает модель, используя .json архитектуру.
    :param path: string
    :param kwargs: dict. При использовании custom_objects
    :return: obj. Keras модель
    '''
    assert os.listdir(path), (f'Директория {path} пуста. Загрузите .json')
    filename = os.path.join(path, filename)
    with open(filename, 'r') as json_file:
        json_structure = json_file.read()
    model = model_from_json(json_structure, kwargs)
    return model


def save_h5_pb(path_pb: str, path_json='', path_h5='', model=None, **kwargs):
    '''
    Восстанавливает архитектуру модели, используя .json.
    Создаёт keras модель в формате .h5
    Конвертирует keras модель из формата .h5 в формат .pb (для TFServ)
    :param model: object. tf.keras
    :param path_json: string
    :param path_h5: string
    :param path_pb: string
    :param kwargs: dict. При использовании custom_objects в tf.keras model
    '''
    if int(__version__.split('.')[0]) == 2:
        if path_json:
            print('load structure.json')
            model = load_structure(path_json, **kwargs)
        if path_h5:
            print('load weights.h5')
            filename = os.path.join(path_h5, 'weights.h5')
            model.load_weights(filename)
        model.save(path_pb)
    else:
        if path_json:
            print('load structure.json')
            model = load_structure(path_json, **kwargs)
        if path_h5:
            print('load weights.h5')
            filename = os.path.join(path_h5, 'weights.h5')
            model.load_weights(filename)
            filename = os.path.join(path_h5, 'model.h5')
            model.save(filename)
        assert len(os.listdir(path_pb)) == 0, (f'Директория {path_pb} не пуста')
        with get_session() as sess:
            simple_save(
                sess,
                path_pb,
                inputs={'input_image': model.input},
                outputs={t.name: t for t in model.outputs})
    clear_session()
    print(f'\nУспешно сохранены данные в {os.path.dirname(path_pb)}')
