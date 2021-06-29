import time
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


IMAGE_SIZE = 224


def gen_train_dataset():

    resize_and_rescale = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    ])

    ds = tfds.load("tf_flowers", split=tfds.Split.TRAIN, data_dir=".tfds")
    ds = ds.map(lambda x: (resize_and_rescale(x["image"], training=True), x["label"]))
    ds = ds.shuffle(1024)
    ds = ds.batch(128)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds


def create_model():

    model_url = "https://tfhub.dev/tensorflow/resnet_50/classification/1"
    model = tf.keras.Sequential([
        hub.KerasLayer(model_url)
    ])
    model.build([None, IMAGE_SIZE, IMAGE_SIZE, 3])

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

    batch_size = 256

    ds_train = gen_train_dataset()

    start = time.time()

    # import IPython;IPython.embed()
    model.fit(
        ds_train,
        epochs=5,
        # steps_per_epoch=int(60000/batch_size),
        # validation_split=0.2,
        # validation_steps=100,
        # batch_size=32,
        verbose=1,
    )

    end = time.time()
    print(f"Time: {end - start} sec")


if __name__ == "__main__":
    main()
