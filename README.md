# Dacon-landmark-recognition
데이콘 랜드마크 분류 AI 경진대회 프로젝트

### 1. TFRecord로 파일 변환
- TFRecord 파일은 Tensorflow의 학습 데이터를 저장하기 위한 Binary data format
- JPG나 PNG 같은 이미지 data를 읽을 때 라벨 데이터를 별도의 csv 파일 등에서 읽거나 모델에서 데이터 처리를 위해 decoding이 필요
- TFRecord 파일로 데이터를 저장하면 라벨과 데이터를 하나의 배치 파일로 저장 가능
- 파일 용량, 학습 속도 면에서 효과
- 아래 코드로 train data set을 700 ~ 800 MB 크기로 나누어서 TFRecord 데이터로 변환
 ```python
if not pht.exists(train_data_path):
  os.system('unzip {}/{} -d {}'.format(data_base_path, train_zip_name, train_data_path))
  place_name_list = [name for name in os.listdir(train_data_path) if not name.endswith('.JPG')]
  for place_name in place_name_list:
    place_fullpath = pth.join(train_data_path, place_name)
    landmark_name_list = os.listdir(place_fullpath)
    for landmark_name in landmark_name_list:
      landmark_fullpath = pth.join(place_fullpath, landmark_name)
      image_name_list = os.listdir(landmark_fullpath)
      for image_name in pth.join(landmark_full, image_name)
      if not image_fullpath.endwith('.JPG'):
        continue
       shutil.move(image_fullpath, train_data_path)
```


### 2. landmark recognition model
- TPU 연동
  - 빠른 이미지 처리를 위해 TPU 사용
  - Strategy API를 사용하여 훈련을 여러 장치로 분산
```python
# tpu 사용 가능 확인
try:
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
  print('Running on TPU', tpu.master())
 except ValueError:   # tpu 사용이 불가능할 땐 gpu 사용
  tpu = None

 if tpu:
  tf.config.experimental_connect_to_cluster(tpu)
  tf.tpu.experimental.initialize_tpu_system(tpu)
  strategy = tf.distribute.TPUStrategy(tpu)
 else:
  # 텐서플로우에서 사용하는 Default Strategy
  strategy = tf.distribute.MirroredStrategy()
```

- 하이퍼파라미터 설정
```python
EPOCHS = 50
BATCH_SIZE = 32
SKIP_VALIDATION = True
IMAGE_SIZE = [540, 960]
RESIZE_SIZE = [600, 600]

INIT_LR = 1e-3
MOMENTUM = 0.9
DECAY = 1e-5
```

- TPU 학습 전략 설정
  - 학습 모델로 EfficientNet 사용
  - 초기 가중치는 **imagenet**으로 학습된 가중치를 사용
  - `enet.trainable()`을 사용하여 model을 학습 이미지에 맞는 가중치로 업데이트
```python
with strategy.scope():
  if PRETRAINED_PATH:
      latest_path = tf.train.latest_checkpoint(PRETRAINED_PATH)
   else:
      latest_path = None
   
   enet = tf.keras.applications.EfficientNetB6(
      input_shape = [*RESIZE_SIZE, 3],
      weights = 'imagenet' if latest_path == None else None,
      include_top = False
   )
   enet.trainable = True
   
   model = tf.keras.Sequential([
      enet,
      tf.keras.layers.GlobalAveragePooling2D(),
      tf.keras.layers.Dense(len(CLASSES)),
      tf.keras.layers.Activation('softmas', dtype='float32')
   )]
   
   if latest_path:
      model.load_weights(latest_path)
   
   model.compile(
      optimizer = tf.keras.optimizers.Adam(lr=0.0001),
      loss = 'sparse_caregorical_crossentropy',
      metrics = ['sparse_categorical_accuracy']
   )
```

- 학습 진행
```python
history = model.fit(
   get_training_dataset(),
   epochs = EPOCHS,
   steps_per_epoch=STEPS_PER_EPOCH,
   callbacks = [model_checkpoint_callback, tensorboard_callback].
   validation_data=get_validation_dataset(),
   validation_steps_VALIDATION_STEP
)
```

- 모델 예측 수행
```python
# 가장 좋은 모델의 가중치 load
checkpoint_path = os.path.join(MODEL_PATH, MODEL_NAME)
weight_file = tf.io.gfile.glob('{}/*.hdf5'.format(checkpoint_path))[-1]
print(weight_file)
model.load_weights(weights_file)

# 예측 수행
test_dataset = tf.data.TFRecordDataset(TEST_FILENAMES)
test_dataset = test_dataset.map(read_labeled_tfrecord).batch(64).prefetch(1)
y_pred = model.predict(test_Dataset)
```

- landmark recognition 결과

![image](https://user-images.githubusercontent.com/72551588/126068666-35ed1217-d52d-4cb7-8bec-6f398cbd7c3a.png)

# Reference
[1] EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks [Submitted on 28 
May 2019 (v1), last revised 11 Sep 2020 (this version, v5)] Mingxing Tan, Quoc V. Le
https://arxiv.org/pdf/1905.11946.pdf

[2] The Marginal Value of Adaptive Gradient Methods in Machine Learning [Submitted on 23 
May 2017 (v1), last revised 22 May 2018 (this version, v2)] Ashia C. Wilson, Rebecca Roelofs, 
Mitchell Stern, Nathan Srebro, Benjamin Recht 
https://arxiv.org/pdf/1705.08292.pdf
