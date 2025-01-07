import os
import time
import glob
import shutil
import pathlib
import argparse
import dataclasses
from pprint import pprint

import mne
import numpy as np
import xarray as xr

from autoreject import AutoReject
from joblib import Parallel, delayed


@dataclasses.dataclass
class CliArguments:
    dataset_path: str
    output_path: str
    # log_output_path: str

    duration: int
    overlap: int

    jobs: int
    temp_dir: str
    debug: bool

    @staticmethod
    def parse(args):
        norm = {k.replace("-", "_"): v for k, v in vars(args).items()}
        return CliArguments(**norm)


# ===========================================================================
# Worker jobs
# ===========================================================================


class PreprocessJob:
    def __init__(self, file_path: str, args: CliArguments) -> None:
        self.file_path = pathlib.Path(file_path)
        self.args = args

    def __call__(self):
        try:
            # read the file
            raw = mne.io.read_raw_edf(
                self.file_path.absolute(), verbose=False, preload=True
            )
            print("PROCESS!   file:", self.file_path.name)

            # update channel and montage
            raw.rename_channels({"FP2": "Fp2"}, verbose=False)
            raw.set_montage("standard_1020", verbose=False)

            # band-pass filter 0.1 Hz - 60 Hz
            raw = raw.filter(l_freq=0.1, h_freq=60, verbose=False)

            # notch filter at 60 Hz
            raw = raw.notch_filter(freqs=(60), verbose=False)

            # create epoch by fixed length
            epochs = mne.make_fixed_length_epochs(
                raw,
                duration=self.args.duration,
                overlap=self.args.overlap,
                preload=True,
                verbose=False,
            )

            if len(epochs) < 5:
                print("SKIPPED:", self.file_path)
                return

            # perform autoreject
            ar = AutoReject(
                n_interpolate=None, random_state=11, n_jobs=1, verbose=False
            )
            _, reject_log = ar.fit_transform(epochs, return_log=True)

            # perform ICA
            ica = mne.preprocessing.ICA(max_iter="auto", random_state=97, verbose=False)
            ica.fit(epochs, verbose=False)
            ica.apply(epochs, verbose=False)

            print(
                "Explained variance by all ICA components: ",
                ica.get_explained_variance_ratio(raw)["eeg"],
            )

            # get the data without bad epochs
            data = epochs[~reject_log.bad_epochs].get_data(copy=True, verbose=False)

            # scale from microvolt to volt
            data = data * 1e6

            # save rejection log
            save_path = (
                pathlib.Path(args.temp_dir) / f"{self.file_path.stem}-rejection_log.npy"
            )
            np.save(save_path.absolute(), reject_log.labels)

            # save signal data
            save_path = pathlib.Path(args.temp_dir) / f"{self.file_path.stem}.npy"
            np.save(save_path.absolute(), data)
        except Exception as e:
            print(f"FAIL!   file:{self.file_path.name}   error: {e}")


# ===========================================================================
# Pipeline jobs
# ===========================================================================


def epoch_data(args: CliArguments):
    print("JOB!   Reading EEG data...")

    # find all EDF files
    files = [file for file in glob.glob(args.dataset_path)]

    # derive data
    if not args.debug:
        jobs = [delayed(lambda x: PreprocessJob(x, args)())(file) for file in files]

        # execute jobs in parallel
        start_time = time.time()
        Parallel(n_jobs=args.jobs)(jobs)
        elapsed_time = time.time() - start_time

        print("Total processing time =", elapsed_time)
    else:
        # execute jobs sequentially
        start_time = time.time()
        for file in files:
            PreprocessJob(file, args)()

        elapsed_time = time.time() - start_time
        print("Total processing time =", elapsed_time)


def merge_data(args: CliArguments):
    print("JOB!   Merging signal dataset...")

    labels = []
    signals = []
    segments = []
    patient_names = []
    recording_numbers = []

    # find all npy files
    search_path = pathlib.Path(args.temp_dir) / "*.npy"
    for file in glob.glob(str(search_path.absolute())):
        # skip rejection log files
        if "rejection_log" in file:
            continue

        # get filename
        filename = pathlib.Path(file)
        name_no_channel = filename.stem.lower()

        # extract metadata from filename
        patient_name = name_no_channel.split("_")[0]
        is_autism = 1 if "autism" in name_no_channel else 0
        rec_number = int(name_no_channel[-2:])

        # load npy
        # (n_segments, n_channels, n_time_step)
        mat = np.load(file)

        # append to list
        labels.extend([is_autism] * mat.shape[0])
        signals.append(mat)
        segments.extend(np.arange(0, mat.shape[0]).tolist())
        patient_names.extend([patient_name] * mat.shape[0])
        recording_numbers.extend([rec_number] * mat.shape[0])

    # convert to numpy
    labels = np.array(labels)
    signals = np.concatenate(signals)
    segments = np.array(segments)
    patient_names = np.array(patient_names)
    recording_numbers = np.array(recording_numbers)

    # move signal to last channel
    signals = np.moveaxis(signals, 1, 2)

    # combine all
    ds = xr.Dataset(
        data_vars={
            "signal": (["samples", "time_steps", "channels"], signals),
            "label": (["samples"], labels),
        },
        coords={
            "samples": range(signals.shape[0]),
            "segment": (["samples"], segments),
            "patient_name": (["samples"], patient_names),
            "recording_number": (["samples"], recording_numbers),
        },
    )

    # show summary
    print(ds)

    # save to file
    ds.to_netcdf(args.output_path)

    # statistics
    df = ds[["label", "patient_name", "samples"]].to_dataframe()
    print("----- SIGNAL DATASET STATISTICS -----")
    print(df.value_counts().sort_index())


def merge_rejecton_logs(args: CliArguments):
    print("JOB!   Merging signal dataset...")

    labels = []
    patient_names = []
    recording_numbers = []
    rejection_logs = []

    # find all npy files
    search_path = pathlib.Path(args.temp_dir) / "*.npy"
    for file in glob.glob(str(search_path.absolute())):
        # skip rejection log files
        if "rejection_log" not in file:
            continue

        # get filename
        filename = pathlib.Path(file)
        name_no_channel = filename.stem.lower().replace("-rejection_log", "")

        # extract metadata from filename
        patient_name = name_no_channel.split("_")[0]
        is_autism = 1 if "autism" in name_no_channel else 0
        rec_number = int(name_no_channel[-2:])

        # load npy
        # (n_time_step, n_channels)
        mat = np.load(file)

        # append to list
        labels.append(is_autism)
        patient_names.append(patient_name)
        recording_numbers.append(rec_number)
        rejection_logs.append(mat)

    # determine biggest shape of the logs
    max_epochs = np.max([x.shape[0] for x in rejection_logs])

    # pad with zeros
    rejection_logs = [
        np.pad(x, pad_width=[(0, max_epochs - x.shape[0]), (0, 0)], mode="constant")
        for x in rejection_logs
    ]

    # convert to numpy
    labels = np.array(labels)
    patient_names = np.array(patient_names)
    recording_numbers = np.array(recording_numbers)
    rejection_logs = np.array(rejection_logs)

    # combine all
    ds = xr.Dataset(
        data_vars={
            "label": (["samples"], labels),
            "rejection_log": (["samples", "epochs", "channels"], rejection_logs),
            "patient_name": (["samples"], patient_names),
            "recording_number": (["samples"], recording_numbers),
        },
        coords={
            "samples": range(rejection_logs.shape[0]),
        },
    )

    # show summary
    print(ds)

    # save to file
    ds.to_netcdf(args.log_output_path)


# ===========================================================================
# Entry point
# ===========================================================================


def main(args: CliArguments):
    # create temp dir
    os.makedirs(args.temp_dir, exist_ok=True)

    # run all jobs
    epoch_data(args)
    merge_data(args)
    # merge_rejecton_logs(args)

    # clean up temp files
    print("CLEAN UP!")
    shutil.rmtree(args.temp_dir)

    print("All clear! File saved at:", args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset-path", type=str)  # ../data/edf/**/*.edf
    parser.add_argument("output-path", type=str)  # ../data/dataset_overlap_60.nc
    # parser.add_argument(
    #     "log-output-path", type=str
    # )  # ../data/dataset_overlap_60_logs.nc

    # epoching
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--overlap", type=int, default=0)

    # parallel jobs
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--temp-dir", type=str, default="./tmp")
    parser.add_argument("--debug", action="store_true")

    # parse CLI
    args = CliArguments.parse(parser.parse_args())
    pprint(args)

    # start the app!
    main(args)
