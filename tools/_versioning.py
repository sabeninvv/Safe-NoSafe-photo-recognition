import os
import hashlib
import yaml
import _conversion
import _analytics


def md5sum(filename, blocksize=65536):
    hash = hashlib.md5()
    with open(filename, "rb") as f:
        for block in iter(lambda: f.read(blocksize), b""):
            hash.update(block)
    return hash.hexdigest()


def create_md5(obj_path):
    md5_str = md5sum(obj_path, blocksize=65536)
    head_obj_path = os.path.dirname(obj_path)
    obj_name = os.path.basename(obj_path)
    obj_name = obj_name.split('.')[1:]
    obj_name = '.'.join(obj_name)
    newfile_name = f'{md5_str}.{obj_name}'
    path_newfile = os.path.join(head_obj_path, newfile_name)
    if not os.path.isfile(path_newfile):
        with open(path_newfile, 'w', encoding='utf-8') as file:
            pass # file.write(md5_str)


def recurc_run(path, file_extensions):
    objects = os.listdir(path)
    for obj_name in objects:
        obj_path = os.path.join(path, obj_name)
        if os.path.isdir(obj_path):
            recurc_run(obj_path, file_extensions)
        else:
            if obj_name.split('.')[-1].split('-')[0] in file_extensions:
                create_md5(obj_path)


def get_md5(path,
            file_extensions=('h5', 'json', 'pb', 'index', 'data')):
    print('The script is running. Wait for completion.')
    recurc_run(path, file_extensions)
    print('Сompleted.')


def save_config(database_dir: str, batch_size: int, epoch: int,
                shape: list, augmentation: bool,
                lr: list, class_weights=None, filename='config.yaml'):
    if not class_weights:
        class_weights = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
    train_params = dict(EPOCH=epoch, BATCH_SIZE=batch_size, CLASS_WEIGHTS=class_weights, LR=lr)
    database_params = dict(DATABASE_DIR=database_dir, SHAPE=shape, AUGMENTATION=augmentation)
    to_yaml = dict(DATABASE_PARAMS=database_params, TRAIN_PARAMS=train_params)
    with open(filename, 'w') as f:
        yaml.dump(to_yaml, f)


def easy_start(path_to_model_version_dirs, model, txt_notification='-'):
    '''
    Создание структуры папок для хранения и версионирования tf.keras model.
    Сохранение архитектуры tf.keras model в .json формате.
    :param path_to_model_version_dirs: str. Абсолютный путь к директории с версиями моделей.
    :param model: object. tf.keras model.
    :param txt_notification: str. Заметки по архитектуре и обучению.
    :return: tuple. Кортеж с абсолютными путями до директории хранения .pb файлов и .h5 файла весов
    '''
    path_h5, path_json, path_config, path_pb = _conversion.make_dirs(path_model_dir=path_to_model_version_dirs)
    _conversion.save_structure(model, path_json=path_json, txt_structure=txt_notification)
    path_weights = os.path.join(path_h5, 'weights.h5')
    return path_pb, path_weights


def easy_finish(path_to_database, dim, model, epoch, lr,
                batch_size, path_pb, augmentation,
                path_to_test_imgs='', path_to_test_vds='', class_weights=None,
                batch_predict=50, frames_count=3, limit=0.7,
                prefix_pred_imgs='FOTO', prefix_pred_vds='VIDEO', git_lfs=True,
                **kwargs):
    '''
    Создание матриц ошибок по видео/фото контенту.
    Создание .yaml файла с описанием.
    :param git_lfs: bool. Dump files with git lfs
    :param frames_count: Количество захватываемых кадров
    :param path_to_test_imgs: string. Абсолютный путь к директории для с изображениями, разбитыми по классам.
    :param path_to_test_vds: string. Абсолютный путь к директории для с видео, разбитыми по классам.
    :param path_to_database: string. Абсолютный путь к базе данных изображений, разбитых по классам.
    :param path_pb: string. Абсолютный путь к директории для хранении .pb файлов.
    :param dim: tuple. Input model
    :param model: object. tf.keras model
    :param epoch: int. Количество эпох обучения.
    :param batch_size: int. Количество batch обучения.
    :param augmentation: bool. Применялась аугментация при обучении.
    :param lr: tuple. Размер шага обучения.
    :param class_weights: dict. Веса классов.
    :param batch_predict: int. Размер batch для model.predict
    :param kwargs: При использовании custom_objects в tf.keras model
    '''
    path_to_maindir = os.path.dirname(path_pb)
    path_to_config = os.path.join(path_to_maindir, 'config')
    path_to_confmtrx = os.path.join(path_to_config, 'confmtrx.md')
    path_to_conf_yaml = os.path.join(path_to_config, 'config.yaml')
    if path_to_test_imgs:
        gen = _analytics.Pred2imgs(path_to_dir=path_to_test_imgs,
                                   model=model, batch=batch_predict, dim=dim,
                                   filename=path_to_confmtrx, prefix=prefix_pred_imgs)
        gen.predict()
    if path_to_test_vds:
        gen = _analytics.Pred2vds(path_to_dir=path_to_test_vds, dim=dim, batch=batch_predict,
                                  model=model, frames_count=frames_count,
                                  filename=path_to_confmtrx, prefix=prefix_pred_vds, limit=limit)
        gen.predict()

    save_config(database_dir=path_to_database, batch_size=batch_size, epoch=epoch,
                shape=list(dim), augmentation=augmentation,
                lr=list(lr), class_weights=class_weights, filename=path_to_conf_yaml)
    _conversion.save_h5_pb(path_pb, model=model, **kwargs)
    if not git_lfs:
        get_md5(path_to_maindir)
