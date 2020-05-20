## Работа с репозиторием
pass

## Инструментарий
* _analytics.py **-->** ```Оценка модели, создание и запись матриц ошибок. Оценка производится пакетом (batch) на фото/видео контенте.```
* _augmentation.py **-->** ```Копирование, изменение размера, искажение/сохранение пропорций фотографии.```
* _conversion.py **-->** ```Версионирование, сохранение, загрузка, конвертирование моделей ```
* _generator.py **-->** ```Генераторы (imgs from storage, RAM, .hdf5) для передачи данных в model.fit().```
* _tools.py **-->** ```Инструменты для предобработки, создания и визуализации данных. ```
* _video.py **-->** ```Декодинрование видео, разбиение на кадры, нормализация данных.```

## Examples
#### _augmentation
* **_augmentation.ImgResizeSave():**
```
class_obj = _augmentation.ImgResizeSave(dir_to_open=r'YOUR_EXISTING_FLODER',
                                        new_dir_to_save=r'YOUR_NEW_FOLDER', 
                                        dim=(HEIGHT, WIDTH, CHANNELS), cut=True, interpolation=cv2.INTER_CUBIC, 
                                        number_of_multiproces=6)
class_obj.imgs_copy()
```

#### _versioning
* _versioning.easy_start(), _versioning.easy_finish()
````python
model = CREATE_MODEL()

path_pb, path_weights = _versioning.easy_start(path_to_model_version_dirs=r'ПУТЬ_ДО_КОРНЕВОГО_КАТАЛОГА_С_ВЕРСИЯМИ_МОДЕЛЕЙ', 
                                               model=model, 
                                               txt_notification='ОПИСАНИЕ_АРХИТЕКТУРЫ_МОДЕЛИ')

######################################################
### В данном блоке происходит обучение модели      ###
### ...                                            ###
### model.fit()                                    ###
### ...                                            ###
######################################################

_versioning.easy_finish(path_to_database=r'ПУТЬ_ДО_БАЗЫ_ДАННЫХ_ДЛЯ_ОБУЧЕНИЯ', 
                        path_to_test_imgs=r'ПУТЬ_ДО_КАТАЛОГА_С_ТЕСТОВЫМИ_ИЗОБРАЖЕНИЯМИ_РАЗБИТЫМИ_НА_КЛАССЫ',
                        path_to_test_vds=r'ПУТЬ_ДО_КАТАЛОГА_С_ТЕСТОВЫМИ_ВИДЕОФАЙЛАМИ_РАЗБИТЫМИ_НА_КЛАССЫ',
                        dim=(128,128,3), model=model, epoch=60, lr=(-3,-4,-5),
                        batch_size=102, path_pb=path_pb, augmentation=True,                        
                        frames_count=10, batch_predict=50, class_weights=None,
                        prefix_pred_imgs='НАЗВАНИЕ_МАТРИЦЫ', prefix_pred_vds='НАЗВАНИЕ_МАТРИЦЫ')
````

#### _generator
* **_generator.ImageGenerator()**
```python
PATH = r'/content/DB_21.04.20_128x128'
BATCH = 120
DIM = (128, 128, 3)

gen_train = generator.ImageGenerator(path_to_dir=PATH,
                                     name_sample='TRAIN',
                                     dim=DIM,
                                     batch=BATCH,
                                     to_categorical=True,
                                     smooth=True)

gen_valid = generator.ImageGenerator(path_to_dir=PATH,
                                     name_sample='VALID',
                                     dim=DIM,
                                     batch=BATCH,
                                     to_categorical=True)

gen_test = generator.ImageGenerator(path_to_dir=PATH,
                                    name_sample='TEST',
                                    dim=DIM,
                                    batch=BATCH,
                                    to_categorical=True)

g_train = gen_train.generator()
g_valid = gen_valid.generator()

history = model.fit(g_train, 
                    epochs=NUM_EPOCHS,
                    steps_per_epoch = gen_train.db_to_gen.shape[0]//BATCH, 
                    validation_data=g_valid,
                    validation_steps=gen_valid.db_to_gen.shape[0]//BATCH,
                    verbose=1)
```
* **_generator.ImageGenerator(). Ручное изменение количества классов.**
```python
# Для примера вы создали объекты класса _generator.ImageGenerator().
# print(gen_train.num_classes)
# -> 6
# Вы страстно желаете схлопнуть до 3 классов.

gen_train.db_to_gen.CLASS[gen_train.db_to_gen.CLASS==3] = 0
gen_train.db_to_gen.CLASS[gen_train.db_to_gen.CLASS==4] = 1
gen_train.db_to_gen.CLASS[gen_train.db_to_gen.CLASS==5] = 0

gen_valid.db_to_gen.CLASS[gen_valid.db_to_gen.CLASS==3] = 0
gen_valid.db_to_gen.CLASS[gen_valid.db_to_gen.CLASS==4] = 1
gen_valid.db_to_gen.CLASS[gen_valid.db_to_gen.CLASS==5] = 0

gen_test.db_to_gen.CLASS[gen_test.db_to_gen.CLASS==3] = 0
gen_test.db_to_gen.CLASS[gen_test.db_to_gen.CLASS==4] = 1
gen_test.db_to_gen.CLASS[gen_test.db_to_gen.CLASS==5] = 0

gen_train.num_classes = 3
gen_valid.num_classes = 3
gen_test.num_classes = 3
```

#### _tools
* **_tools.createZeros(), _tools.fillZeros()**
```python
# Для примера вы хотите выгрузить и нормализовать данные, хранящиеся в объекте класса _generator.ImageGenerator().

X_test, y_test = tools.createZeros(db=gen_test.db_to_gen, 
                                   dim=(128, 128, 3))

X_test, y_test = tools.fillZeros(X=X_test, y=y_test, 
                                 dim=(128, 128, 3), 
                                 link_db=gen_test.db_to_gen)
```

* **_tools.one_epoch_confmtrx()**
```python
model, X_test, y_test, API_KEY, NAME_APP = None, None, None, None, None
API_KEY = YOUR_API_KEY_IFTTT
NAME_APP =  YOUR_APP_NAME_IFTTT

# В функцию, как и в калбэк будут падаваться ССЫЛКИ на переменные: model, X_test, y_test, API_KEY, NAME_APP 
def one_epoch_confmtrx(epoch, logs):
    print()
    indxs = ['epoch', 'logs', 'model', 'API_KEY', 'NAME_APP', 'X_test', 'y_test']
    val = [epoch, logs, model, API_KEY, NAME_APP, X_test, y_test]
    kwargs = dict(zip(indxs, val))
    _tools.one_epoch_confmtrx(**kwargs)

printConfMtrx = LambdaCallback(on_epoch_end=one_epoch_confmtrx)
```

#### _conversion
* **_conversion.make_dirs(), _conversion.save_structure**
```python
model = CREATE_YOUR_MODEL()

path2model_name = r'/home/jovyan/ai/models/classification/dnn'

txt_notification = '''
YOUR_Description
YOUR_Config
'''

path_json, path_h5, path_pb = _conversion.make_dirs(path2model_name)
_conversion.save_structure(model=model, path_json=path_json, txt_structure=txt_notification)
weights_file = os.path.join(path_h5, 'weights.h5')
```
* **_conversion.load_structure()**
```python
model = _conversion.load_structure(path='/home/jovyan/ai/models/classification/dnn/12/json', 
                                   filename='structure.json',
                                   relu6=relu6) # **kwargs
model.load_weights(weights_file)
```

#### _analytics
* **_analytics.Pred2imgs(), _analytics.Pred2vds()**
```python
# For IMAGES
gen = _analytics.Pred2imgs(path_to_dir=r'YOUR_PATH', quantiles=(.55, .65, .75, .85, .95),
                           model=model, batch=100, dim=(128, 128, 3),
                           interpolation=cv2.INTER_CUBIC, resize=True, normalize=True, 
                           filename='PATH_TO_FILE.md')
gen.predict()

# For VIDEOS
gen = _analytics.Pred2vds(path_to_dir=r'YOUR_PATH', dim=(128,128,3), batch=100, 
                          quantiles=(.55, .65, .75, .85, .95), 
                          model=model, only_key_frames=False, frames_count=3,
                          resize=True, normalize=True,
                          interpolation=cv2.INTER_CUBIC, filename='PATH_TO_FILE.md')
gen.predict()
```

#### _video
* **_video.video_to_arrays()**
```python
path = r'PATH_TO_YOUR_VIDEOFILE'
arrs = _video.video_to_arrays(path=path, only_key_frames=True, frames_count=100,
                              interpolation=cv2.INTER_CUBIC, resize=True, normalize=True)
```