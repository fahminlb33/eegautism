import random
from enum import Enum

from netCDF4 import Dataset
import xarray as xr
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

GLOBAL_RANDOM_SEED = 42


class ArchitectureEnum(Enum):
    # tensorflow
    EEG_NET = "eegnet"
    DEEP_SLEEP_NET = "deepsleepnet"
    CHRONO_NET = "chrononet"
    FILIP_NET = "filipnet"
    VANILLA_CNN_1D = "vanillacnn1d"


# ----------------------------------------------------------------------------
#  TENSORFLOW MODELS
# ----------------------------------------------------------------------------


# EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces
def EEGNet(n_channels, n_timestep, n_kernels=1):
    # (n_samples, n_channels, n_timestep, n_kernels)
    inputs = tf.keras.layers.Input(shape=(n_channels, n_timestep, n_kernels))

    # block 1
    x = tf.keras.layers.Conv2D(8, kernel_size=(1, 32), padding="same", use_bias=False)(
        inputs
    )
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(n_channels, 1),
        use_bias=False,
        depth_multiplier=2,
        depthwise_constraint=tf.keras.constraints.max_norm(1.0),
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("elu")(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(1, 4))(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # block 2
    x = tf.keras.layers.SeparableConv2D(
        16, kernel_size=(1, 16), padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("elu")(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(1, 8))(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Flatten()(x)

    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid", kernel_constraint=tf.keras.constraints.max_norm(0.25)
    )(x)

    # connect model graph
    return tf.keras.Model(inputs=inputs, outputs=outputs)


# DeepSleepNet: a Model for Automatic Sleep Stage Scoring based on Raw Single-Channel EEG
def DeepSleepNet(n_timestep, n_channels):
    # (n_samples, n_timestep, n_channels)
    inputs = tf.keras.layers.Input(shape=(n_timestep, n_channels))

    # representation learning
    # high-freq
    x1 = tf.keras.layers.Conv1D(
        64, kernel_size=256 // 2, strides=256 // 16, padding="same"
    )(inputs)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Activation("relu")(x1)

    x1 = tf.keras.layers.MaxPooling1D(pool_size=8, strides=8)(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)

    x1 = tf.keras.layers.Conv1D(128, kernel_size=8, padding="same")(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Activation("relu")(x1)

    x1 = tf.keras.layers.Conv1D(128, kernel_size=8, padding="same")(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Activation("relu")(x1)

    x1 = tf.keras.layers.Conv1D(128, kernel_size=8, padding="same")(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Activation("relu")(x1)

    x1 = tf.keras.layers.MaxPooling1D(pool_size=4, strides=4)(x1)

    # low-freq
    x2 = tf.keras.layers.Conv1D(
        64, kernel_size=256 * 4, strides=256 // 2, padding="same"
    )(inputs)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Activation("relu")(x2)

    x2 = tf.keras.layers.MaxPooling1D(pool_size=4, strides=4)(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)

    x2 = tf.keras.layers.Conv1D(128, kernel_size=6, padding="same")(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Activation("relu")(x2)

    x2 = tf.keras.layers.Conv1D(128, kernel_size=6, padding="same")(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Activation("relu")(x2)

    x2 = tf.keras.layers.Conv1D(128, kernel_size=6, padding="same")(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Activation("relu")(x2)

    x2 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)(x2)

    xt = tf.keras.layers.Concatenate(axis=1)([x1, x2])

    # sequence residual learning
    xd = tf.keras.layers.Dropout(0.5)(xt)

    x3 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(512, return_sequences=False)
    )(xd)
    x3 = tf.keras.layers.Dropout(0.5)(x3)
    x3 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(512, return_sequences=False)
    )(xd)
    x3 = tf.keras.layers.Dropout(0.5)(x3)

    x4 = tf.keras.layers.Dense(1024, activation="relu")(xd)

    cs = tf.keras.layers.Add()([x3, x4])
    x = tf.keras.layers.Dropout(0.5)(cs)

    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    # connect model graph
    return tf.keras.Model(inputs=inputs, outputs=outputs)


# ChronoNet: A Deep Recurrent Neural Network for Abnormal EEG Identification
def ChronoNet(n_timestep, n_channels):
    # (n_samples, n_timestep, n_channels)
    inputs = tf.keras.layers.Input(shape=(n_timestep, n_channels))

    # feature map
    x1 = tf.keras.layers.Conv1D(32, kernel_size=2, strides=2, padding="same")(inputs)
    x2 = tf.keras.layers.Conv1D(32, kernel_size=4, strides=2, padding="same")(inputs)
    x3 = tf.keras.layers.Conv1D(32, kernel_size=8, strides=2, padding="same")(inputs)

    c = tf.keras.layers.Concatenate(axis=2)([x1, x2, x3])

    x1 = tf.keras.layers.Conv1D(32, kernel_size=2, strides=2, padding="same")(c)
    x2 = tf.keras.layers.Conv1D(32, kernel_size=4, strides=2, padding="same")(c)
    x3 = tf.keras.layers.Conv1D(32, kernel_size=8, strides=2, padding="same")(c)

    c = tf.keras.layers.Concatenate(axis=2)([x1, x2, x3])

    x1 = tf.keras.layers.Conv1D(32, kernel_size=2, strides=2, padding="same")(c)
    x2 = tf.keras.layers.Conv1D(32, kernel_size=4, strides=2, padding="same")(c)
    x3 = tf.keras.layers.Conv1D(32, kernel_size=8, strides=2, padding="same")(c)

    c = tf.keras.layers.Concatenate(axis=2)([x1, x2, x3])

    # recurrent
    g1 = tf.keras.layers.GRU(32, return_sequences=True)(c)
    g2 = tf.keras.layers.GRU(32, return_sequences=True)(g1)
    cg1 = tf.keras.layers.Concatenate(axis=2)([g1, g2])

    g3 = tf.keras.layers.GRU(32, return_sequences=True)(cg1)
    cg2 = tf.keras.layers.Concatenate(axis=2)([g1, g2, g3])

    g4 = tf.keras.layers.GRU(32, return_sequences=False)(cg2)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(g4)

    # connect model graph
    return tf.keras.Model(inputs=inputs, outputs=outputs)


# Recurrent and Convolutional Neural Networks in Classification of EEG Signal for Guided Imagery and Mental Workload Detection
def FilipNet(n_timestep, n_channels):
    # (n_samples, n_timestep, n_channels)
    inputs = tf.keras.layers.Input(shape=(n_timestep, n_channels))

    x = tf.keras.layers.Conv1D(
        16, kernel_size=3, strides=2, padding="same", use_bias=False
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("leaky_relu")(x)

    x = tf.keras.layers.Conv1D(
        16, kernel_size=3, strides=2, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("leaky_relu")(x)

    x = tf.keras.layers.SpatialDropout1D(0.25)(x)

    x = tf.keras.layers.Conv1D(
        64, kernel_size=3, strides=2, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.Activation("leaky_relu")(x)

    x = tf.keras.layers.Conv1D(
        128, kernel_size=3, strides=2, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("leaky_relu")(x)

    # skip TimeDistributed
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32 * 2))(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    # connect model graph
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def VanillaCNN1D(n_timestep, n_channels):
    inputs = tf.keras.layers.Input(
        shape=(
            n_timestep,
            n_channels,
        )
    )

    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding="same")(inputs)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)

    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    # x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


# ----------------------------------------------------------------------------
#  DATA LOADING AND MODEL
# ----------------------------------------------------------------------------


# for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data(arch: ArchitectureEnum, path: str, test_size=0, test_patients=[]):
    # load dataset
    dataset = xr.open_dataset(path)
    df = dataset[["label", "patient_name", "samples"]].to_dataframe()

    # --- SPLIT METHOD
    if len(test_patients) > 0:
        print("---> USING TEST PATIENTS SPLIT")

        # split train and test persons
        test_dataset = dataset.sel(
            samples=df[df["patient_name"].isin(test_patients)].index.tolist()
        )
        train_dataset = dataset.sel(
            samples=df[~df["patient_name"].isin(test_patients)].index.tolist()
        )

        # get Xy for train
        X_train = train_dataset["signal"].to_numpy()
        y_train = train_dataset["label"].to_numpy()

        # get Xy for test
        X_test = test_dataset["signal"].to_numpy()
        y_test = test_dataset["label"].to_numpy()
    else:
        print("---> USING STRATIFIED RANDOM SPLIT")

        # get Xy
        X = dataset["signal"].to_numpy()
        y = dataset["label"].to_numpy()

        # stratified random sampling
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

    # print data statistics
    print("Train:", X_train.shape, y_train.shape)
    print("Classdist:", np.unique(y_train, return_counts=True))

    print("Test:", X_test.shape, y_test.shape)
    print("Classdist:", np.unique(y_test, return_counts=True))

    # reshape, if necessary
    if arch == ArchitectureEnum.EEG_NET:
        # (n_samples, n_timestep, n_channels) ---> (n_samples, n_channels, n_timestep, n_kernels)
        X_train = np.expand_dims(np.moveaxis(X_train, 2, 1), axis=-1)
        X_test = np.expand_dims(np.moveaxis(X_test, 2, 1), axis=-1)

    return X_train, X_test, y_train, y_test


def create_model(arch: ArchitectureEnum, X_train: np.ndarray, learning_rate: float):
    # create model
    if arch == ArchitectureEnum.EEG_NET:
        arch = EEGNet(X_train.shape[1], X_train.shape[2], X_train.shape[3])
    elif arch == ArchitectureEnum.DEEP_SLEEP_NET:
        arch = DeepSleepNet(X_train.shape[1], X_train.shape[2])
    elif arch == ArchitectureEnum.CHRONO_NET:
        arch = ChronoNet(X_train.shape[1], X_train.shape[2])
    elif arch == ArchitectureEnum.FILIP_NET:
        arch = FilipNet(X_train.shape[1], X_train.shape[2])
    elif arch == ArchitectureEnum.VANILLA_CNN_1D:
        arch = VanillaCNN1D(X_train.shape[1], X_train.shape[2])
    else:
        raise ValueError("Unknown model")

    # compile model
    arch.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="prc", curve="PR"),  # precision-recall curve
        ],
    )

    return arch
