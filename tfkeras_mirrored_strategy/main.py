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

    num_classes = 5

    model_url = "https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/5"
    model = tf.keras.Sequential([
        hub.KerasLayer(model_url),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
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
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, nesterov=True),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

    ds_train = gen_train_dataset()

    start = time.time()

    model.fit(
        ds_train,
        epochs=100,
        verbose=2,
    )

    end = time.time()
    print(f"Time: {end - start} sec")


if __name__ == "__main__":
    main()
