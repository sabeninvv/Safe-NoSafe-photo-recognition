```mermaid
graph TB
  subgraph "Структура хранения версионированных данных"
  SubGraph1["https://git.coolrocket.com/coolrocket/ai/DEEP_LEARNING_TASK/MODEL_NAME/MODEL_VERSION_NUMER/"]
  end
  
  SubGraph1 --- SubGraph1Flow1
  subgraph " "
  SubGraph1Flow1(pb/)
  SubGraph1Flow1 --- variables/ --- variables.data
  variables/ --- variables.index
  SubGraph1Flow1 --- saved_model.pb
  end
  
  SubGraph1 --- SubGraph1Flow2
  subgraph " "
  SubGraph1Flow2(config/)
  SubGraph1Flow2 --- config.yaml
  SubGraph1Flow2 --- confmatrix.md
  end
  
  SubGraph1 --- SubGraph1Flow3
  subgraph " "
  SubGraph1Flow3(json/)
  SubGraph1Flow3 --- strucure.json
  end
 
  SubGraph1 --- SubGraph1Flow4
  subgraph " "
  SubGraph1Flow4(h5/)
  SubGraph1Flow4 --- model.h5
  SubGraph1Flow4 --- weights.h5
  end

```

### Сериализация файлов tensorflow.keras модели:
Директории: **h5, json/**

* weights.h5 - веса модели
* structure.json - архитектура модели
* model.h5 = weights.h5 + structure.json

### Данные для оценки и  повторяемости обучения:
Директория: **config/**
* .yaml = dataset_param + train_param + model_param
* confmatrix.md - матрицы ошибок (test_foto, test_video)

### Сериализация файлов tensorflow.keras модели под TensorFlow Serving:
Директория: **pb/**

* saved_model.pb - веса модели
* variables.data - архитектура модели
* variables.index - архитектура модели
