import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import json
import time
import argparse
import dataclasses

from eeg import ArchitectureEnum, set_seeds, load_data, create_model

import matplotlib
import tensorflow as tf

from matplotlib.figure import Figure
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

@dataclasses.dataclass
class CliArguments:
    dataset_file: str
    output_path: str

    name: str
    epochs: int
    batch_size: int
    test_size: float
    test_patients: list[str]
    arch: ArchitectureEnum

    @staticmethod
    def parse(args):
        norm = {k.replace("-", "_"): v for k, v in vars(args).items()}
        if "test_patients" in norm and norm["test_patients"] is not None:
            norm["test_patients"] = norm["test_patients"].split(",")
        else:
            norm["test_patients"] = []

        norm["arch"] = ArchitectureEnum(args.arch)

        return CliArguments(**norm)


# ----------------------------------------------------------------------------
#  ENTRY POINT
# ----------------------------------------------------------------------------


def main(args: CliArguments):
    # load dataset
    X_train, X_test, y_train, y_test = load_data(
        args.arch, args.dataset_file, args.test_size, args.test_patients
    )

    # create model
    model = create_model(args.arch, X_train)
    print(model.summary())

    # create callbacks
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )

    tb = tf.keras.callbacks.TensorBoard(
        histogram_freq=1,
        log_dir=f"{args.output_path}/tensorboard/{args.name}",
    )

    # fit model
    train_start = time.time()
    model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[es, tb],
        verbose=2,
    )
    train_elapsed = time.time() - train_start

    # run prediction
    test_start = time.time()
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    test_elapsed = time.time() - test_start

    # print eval
    print(classification_report(y_test, y_pred))

    # evaluate model
    with open(f"{args.output_path}/metrics-eval.jsonl", "a+") as f:
        cm = confusion_matrix(y_test, y_pred)

        json.dump(
            {
                "name": args.name,
                "arch": args.arch.value,
                "dataset": args.dataset_file,
                "accuracy": accuracy_score(y_test, y_pred),
                "mcc": matthews_corrcoef(y_test, y_pred),
                "roc-auc": roc_auc_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "tn": float(cm[0, 0]),
                "fn": float(cm[1, 0]),
                "tp": float(cm[1, 1]),
                "fp": float(cm[0, 1]),
                "train_elapsed": train_elapsed,
                "test_elapsed": test_elapsed,
            },
            f,
        )

        f.write("\n")
        f.flush()

    # plot confusion matrix
    fig = Figure()
    ax = fig.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    fig.savefig(os.path.join(args.output_path, f"{args.name}-confusion_matrix.png"))

    # plot ROC
    fig = Figure()
    ax = fig.subplots()
    RocCurveDisplay.from_predictions(y_test, y_pred, ax=ax)
    fig.savefig(os.path.join(args.output_path, f"{args.name}-roc_curve.png"))

    # plot precision-recall
    fig = Figure()
    ax = fig.subplots()
    PrecisionRecallDisplay.from_predictions(y_test, y_pred, ax=ax)
    fig.savefig(os.path.join(args.output_path, f"{args.name}-precision_recall.png"))


if __name__ == "__main__":
    # for reproducibility
    set_seeds()
    matplotlib.use("Agg")

    # create CLI parser
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset-file", type=str)
    parser.add_argument("output-path", type=str)

    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-size", type=float)
    parser.add_argument("--test-patients", type=str)
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=[x.value for x in ArchitectureEnum],
    )

    # parse CLI
    args = CliArguments.parse(parser.parse_args())
    print(repr(args))

    # start the app!
    main(args)
