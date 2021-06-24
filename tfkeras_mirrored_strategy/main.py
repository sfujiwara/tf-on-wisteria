import os
import time
import tensorflow as tf
import tensorflow_datasets as tfds


def train_dataset(batch_size: int):

    ds = tfds.load("mnist", split=tfds.Split.TRAIN, data_dir=".tfds")
    ds = ds.map(lambda x: (tf.cast(x["image"], tf.float32)/255., x["label"]))
    ds = ds.repeat().shuffle(batch_size*2).batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds


def eval_dataset():

    ds = tfds.load("mnist", split=tfds.Split.TEST, data_dir=".tfds")
    ds = ds.map(lambda x: (tf.cast(x["image"], tf.float32)/255., x["label"]))
    ds = ds.repeat().batch(100)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds


def create_model():

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(rate=0.75),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    return model


def main():

    start = time.time()

    # DistributeStrategy for single-node / multi-gpus
    dist_strategy = tf.distribute.MirroredStrategy()

    # Build model in MirroredStrategy scope
    with dist_strategy.scope():
        model = create_model()
        model.compile(
            optimizer=tf.keras.optimizers.SGD(lr=0.01, nesterov=True),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

    batch_size = 2048

    ds_train = train_dataset(batch_size=batch_size)
    ds_eval = eval_dataset()

    start = time.time()

    model.fit(
        ds_train,
        epochs=5,
        steps_per_epoch=int(60000/batch_size),
        validation_data=ds_eval,
        validation_steps=100,
        verbose=2,
    )

    end = time.time()
    print(f"Time: {end - start} sec")


if __name__ == "__main__":
    main()