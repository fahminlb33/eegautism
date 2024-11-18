import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import json
import time
import argparse
import dataclasses

from eeg import ArchitectureEnum, set_seeds, load_data, create_model

import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    matthews_corrcoef,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


GLOBAL_RANDOM_SEED = 42


@dataclasses.dataclass
class CliArguments:
    dataset_file: str
    metrics_file: str

    name: str
    cv: int
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
    X, _, y, _ = load_data(
        args.arch, args.dataset_file, args.test_size, args.test_patients
    )

    # create CV split
    cv = StratifiedKFold(
        shuffle=True,
        n_splits=args.cv,
        random_state=GLOBAL_RANDOM_SEED,
    )

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        print(f"Training fold {fold_i + 1}")

        # split data
        X_train, X_test = (
            X[train_idx],
            X[test_idx],
        )
        y_train, y_test = (
            y[train_idx],
            y[test_idx],
        )

        # create model
        model = create_model(args.arch, X_train)

        # create callbacks
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )

        # fit model
        train_start = time.time()
        model.fit(
            X_train,
            y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=[es],
            verbose=2,
        )
        train_elapsed = time.time() - train_start

        # run prediction
        test_start = time.time()
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        test_elapsed = time.time() - test_start

        # evaluate model
        # write result to csv
        with open(args.metrics_file, "a+") as f:
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


if __name__ == "__main__":
    # for reproducibility
    set_seeds()

    # create CLI parser
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset-file", type=str)
    parser.add_argument("metrics-file", type=str)

    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--cv", type=int, default=10)
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
