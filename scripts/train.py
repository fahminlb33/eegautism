import os

os.environ['PYTHONHASHSEED'] = str(42)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import random
import argparse
import dataclasses
from pprint import pprint

from netCDF4 import Dataset
import numpy as np
import xarray as xr

import pywt

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    matthews_corrcoef,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from models import EEGNet, DeepSleepNet, ChronoNet, FilipNet, VanillaCNN1D


@dataclasses.dataclass
class CliArguments:
    dataset_path: str
    output_path: str

    name: str
    model: str
    epochs: int
    test_size: float
    test_patients: list[str]

    @staticmethod
    def parse(args):
        norm = {k.replace("-", "_"): v for k, v in vars(args).items()}
        if "test_patients" in norm and norm["test_patients"] is not None:
            norm["test_patients"] = norm["test_patients"].split(",")
        else:
            norm["test_patients"] = []

        return CliArguments(**norm)

# ===========================================================================
# Data loading and preprocessing
# ===========================================================================

# for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_data(args: CliArguments):
    # load dataset
    dataset = xr.open_dataset(args.dataset_path)
    df = dataset[["label", "patient_name", "samples"]].to_dataframe()

    print(df.value_counts().sort_index())

    # --- SPLIT METHOD
    if len(args.test_patients) > 0:
        print("---> USING TEST PATIENTS SPLIT")
        
        # split train and test persons
        test_dataset = dataset.sel(
            samples=df[df["patient_name"].isin(args.test_patients)].index.tolist()
        )
        train_dataset = dataset.sel(
            samples=df[~df["patient_name"].isin(args.test_patients)].index.tolist()
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=42)

    # print data statistics
    print("Train:", X_train.shape, y_train.shape)
    print("Classdist:", np.unique(y_train, return_counts=True))

    print("Test:", X_test.shape, y_test.shape)
    print("Classdist:", np.unique(y_test, return_counts=True))

    # reshape, if necessary
    if args.model == "eegnet":
        # (n_samples, n_timestep, n_channels) ---> (n_samples, n_channels, n_timestep, n_kernels)
        X_train = np.expand_dims(np.moveaxis(X_train, 2, 1), axis=-1)
        X_test = np.expand_dims(np.moveaxis(X_test, 2, 1), axis=-1)
        print("RESHAPED FOR EEGNET:", X_train.shape, X_test.shape)

    return X_train, X_test, y_train, y_test


def create_model(args: CliArguments, X_train: np.ndarray):
    # create model
    if args.model == "eegnet":
        model = EEGNet(X_train.shape[1], X_train.shape[2], X_train.shape[3])
    elif args.model == "deepsleepnet":
        model = DeepSleepNet(X_train.shape[1], X_train.shape[2])
    elif args.model == "chrononet":
        model = ChronoNet(X_train.shape[1], X_train.shape[2])
    elif args.model == "filipnet":
        model = FilipNet(X_train.shape[1], X_train.shape[2])
    elif args.model == "vanillacnn1d":
        model = VanillaCNN1D(X_train.shape[1], X_train.shape[2])
    else:
        raise ValueError("Unknown model")

    # compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        # loss=tf.keras.losses.BinaryFocalCrossentropy(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
              tf.keras.metrics.Precision(name='precision'),
              tf.keras.metrics.Recall(name='recall'),
              tf.keras.metrics.AUC(name='auc'),
              tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
        ],
    )

    return model


def write_plot(args: CliArguments, model):
    mp = tf.keras.utils.model_to_dot(
        model, show_layer_names=True, show_shapes=True, dpi=100
    )

    with open(f"{args.output_path}/{args.name}_plot.graphviz", "w") as f:
        f.write(mp.to_string())


def write_eval(args: CliArguments, y_test, y_pred, train_duration):
    # print results
    print("Accuracy:", np.round(accuracy_score(y_test, y_pred), 4))
    print("MCC:", np.round(matthews_corrcoef(y_test, y_pred), 4))
    print("ROC-AUC:", np.round(roc_auc_score(y_test, y_pred), 4))
    print("Precision:", np.round(precision_score(y_test, y_pred), 4))
    print("Recall:", np.round(recall_score(y_test, y_pred), 4))
    print("F1:", np.round(f1_score(y_test, y_pred), 4))
    print(classification_report(y_test, y_pred))

    # write result to csv
    with open(f"{args.output_path}/{args.name}.csv", "w") as f:
        cm = confusion_matrix(y_test, y_pred)

        f.write("accuracy,mcc,roc_auc,precision,recall,f1,tn,fn,tp,fp,duration_sec\n")
        f.write(f"{accuracy_score(y_test, y_pred)},")
        f.write(f"{matthews_corrcoef(y_test, y_pred)},")
        f.write(f"{roc_auc_score(y_test, y_pred)},")
        f.write(f"{precision_score(y_test, y_pred)},")
        f.write(f"{recall_score(y_test, y_pred)},")
        f.write(f"{f1_score(y_test, y_pred)},")
        f.write(f"{cm[0, 0]},")  # tn
        f.write(f"{cm[1, 0]},")  # fn
        f.write(f"{cm[1, 1]},")  # tp
        f.write(f"{cm[0, 1]},")  # fp
        f.write(f"{train_duration}")


# ===========================================================================
# Entry point
# ===========================================================================


def main(args: CliArguments):
    # load dataset
    X_train, X_test, y_train, y_test = load_data(args)

    # create tensorflow dataset
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(X_train.shape[0])
        .batch(16)
    )
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(16)

    # create model
    model = create_model(args, X_train)

    # save to dot
    write_plot(args, model)

    # create callbacks
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
    )
    tb = tf.keras.callbacks.TensorBoard(
        log_dir=f"{args.output_path}/logs/{args.name}", histogram_freq=1
    )

    # fit model
    start_time = time.time()
    model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=test_ds,
        callbacks=[tb, es],
    )
    train_duration = time.time() - start_time

    # save model
    model.save(f"{args.output_path}/{args.name}.keras")

    # run prediction
    y_pred = (model.predict(test_ds) > 0.5).astype(int)

    # evaluate model
    write_eval(args, y_test, y_pred, train_duration)


if __name__ == "__main__":
    # for reproducibility
    set_seeds()

    # create CLI parser
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset-path", type=str)  # ../data/cwt.nc
    parser.add_argument("output-path", type=str)  # ../data/models

    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "eegnet",
            "deepsleepnet",
            "chrononet",
            "filipnet",
            "vanillacnn1d",
        ],
        default="vanillacnn1d",
    )
    parser.add_argument("--test-size", type=float)
    parser.add_argument("--test-patients", type=str)

    # parse CLI
    args = CliArguments.parse(parser.parse_args())
    pprint(repr(args))

    # start the app!
    main(args)
