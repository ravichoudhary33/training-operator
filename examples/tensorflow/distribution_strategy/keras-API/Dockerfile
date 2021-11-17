FROM tensorflow/tensorflow:2.1.0-gpu-py3

RUN pip install tensorflow_datasets==2.1.0

COPY multi_worker_strategy-with-keras.py /
COPY data/x_train.npy /
COPY data/x_test.npy /
COPY data/y_train.npy /
COPY data/y_test.npy /

ENTRYPOINT ["python", "/multi_worker_strategy-with-keras.py", "--saved_model_dir", "/train/saved_model/", "--checkpoint_dir", "/train/checkpoint"]
