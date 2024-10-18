import os
import re
import time
import glob
import shutil
import pathlib
import argparse
import dataclasses

from io import BytesIO
from pprint import pprint
from collections import Counter

import mne
import pywt
import matplotlib
import scipy.stats
import numpy as np
import xarray as xr

from PIL import Image
from autoreject import AutoReject
from matplotlib.figure import Figure
from joblib import Parallel, delayed


@dataclasses.dataclass
class CliArguments:
    dataset_path: str
    output_path: str

    low_pass: bool
    ica: bool
    auto_reject: bool
    duration: int
    overlap: int

    wavelet: str
    feat_stats: bool
    feat_entropy: bool

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


def scale(x):
    return x * 1e6


def calculate_statistics(list_values):
    return [
        np.nanpercentile(list_values, 5),
        np.nanpercentile(list_values, 25),
        np.nanpercentile(list_values, 75),
        np.nanpercentile(list_values, 95),
        np.nanpercentile(list_values, 50),
        np.nanmean(list_values),
        np.nanstd(list_values),
        np.nanvar(list_values),
        np.nanmean(np.sqrt(list_values**2)),
    ]


def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1] / len(list_values) for elem in counter_values]

    threshold_entropy = 0.2
    renyi_alpha = 2

    return [
        # log energy
        # np.sum(np.log(np.square(probabilities))),
        # threshold entropy
        # np.sum(np.where(np.array(probabilities) > threshold_entropy, 1, 0)),
        # renyi entropy
        # np.log(np.sum(np.power(probabilities, renyi_alpha))) * (1 / renyi_alpha),
        # shannon
        scipy.stats.entropy(probabilities),
    ]


class PreprocessJob:
    def __init__(self, file_path: str, args: CliArguments) -> None:
        self.file_path = pathlib.Path(file_path)
        self.args = args

        self.segments = None
        self.rejection_logs = []

    def __call__(self):
        # read segments
        self.read_segments()

        # check if we successfully read segment, if not, return
        if self.segments is None:
            return

        # process each segment
        processed = []
        for i in range(self.segments.shape[0]):
            print("   file:", self.file_path.name, "   segment:", i)

            if args.wavelet == "mexh":
                processed.append(self.preprocess_cwt(self.segments[i, :, :]))
            elif args.wavelet == "db4" or args.wavelet == "bior2.6":
                processed.append(
                    self.preprocess_dwt_stats_entropy(self.segments[i, :, :])
                )
            else:
                processed.append(self.segments[i, :, :])

        # save rejection log
        if len(self.rejection_logs) > 1:
            save_path = (
                pathlib.Path(args.temp_dir)
                / f"{self.file_path.stem}-rejection_logs.npy"
            )
            np.save(save_path.absolute(), np.array(self.rejection_logs))

        # save npy
        save_path = pathlib.Path(args.temp_dir) / f"{self.file_path.stem}.npy"
        np.save(save_path.absolute(), np.array(processed))

    def read_segments(self):
        try:
            # read the file
            raw = mne.io.read_raw_edf(
                self.file_path.absolute(), verbose=False, preload=True
            )
            print("PROCESS!   file:", self.file_path.name)

            # update channel and montage
            raw.rename_channels({"FP2": "Fp2"}, verbose=False)
            raw.set_montage("standard_1005", verbose=False)

            # hi-pass
            if self.args.low_pass:
                raw = raw.filter(l_freq=1.0, h_freq=None, verbose=False)

            # create epoch by fixed length
            epochs = mne.make_fixed_length_epochs(
                raw,
                duration=self.args.duration,
                overlap=self.args.overlap,
                preload=True,
                verbose=False,
            )

            # perform autoreject
            if self.args.auto_reject:
                ar = AutoReject(
                    n_interpolate=[1, 2, 3, 4], random_state=11, n_jobs=1, verbose=False
                )
                epochs, reject_log = ar.fit_transform(epochs, return_log=True)

                self.rejection_logs.append(reject_log.labels)

            # perform ICA
            if self.args.ica:
                ica = mne.preprocessing.ICA(
                    max_iter="auto", random_state=97, verbose=False
                )
                ica.fit(epochs, verbose=False)
                ica.apply(epochs, verbose=False)

                print(
                    "Explained variance by all ICA components: ",
                    ica.get_explained_variance_ratio(raw)["eeg"],
                )

            # get the segments
            self.segments = epochs.get_data(verbose=False, copy=True)
        except Exception as e:
            print(f"FAIL!   file:{self.file_path.name}   error: {e}")

    def preprocess_cwt(self, segment: np.ndarray):
        # perform CWT
        coeff, _ = pywt.cwt(segment, range(1, 128), args.wavelet, 1)

        # reshape the CWT
        return coeff.reshape(127, -1, self.segments.shape[1])[:, :127, :]

    def preprocess_dwt_stats_entropy(self, segment: np.ndarray):
        # to store all data
        features = []

        # for each channel in the input data,
        for c in range(self.segments.shape[1]):
            # perform decomposition
            decomposed = pywt.wavedec(segment[c, :], self.args.wavelet, level=5)

            # derive stats/entropy data from all of the decomposition
            for decom in decomposed:
                # extract statistical features
                if self.args.feat_stats:
                    features += calculate_statistics(decom)

                # extract statistical features
                if self.args.feat_entropy:
                    features += calculate_entropy(decom)

        return np.array(features)


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


def merge(args: CliArguments):
    print("JOB!   Merging features...")

    labels = []
    signals = []
    segments = []
    patient_names = []
    recording_numbers = []

    # find all npy files
    search_path = pathlib.Path(args.temp_dir) / "*.npy"
    for file in glob.glob(str(search_path.absolute())):
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
        print(mat.shape)

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

    print("labels", labels.shape)
    print("signals", signals.shape)
    print("segments", segments.shape)
    print("patient_names", patient_names.shape)
    print("recording_numbers", recording_numbers.shape)

    # set coordinates
    signal_coords = ["samples", "time_steps", "channels"]
    if len(signals.shape) == 2:
        signal_coords = ["samples", "features"]
    elif len(signals.shape) == 3:
        # move signal to last channel
        signals = np.moveaxis(signals, 1, 2)

    # combine all
    ds = xr.Dataset(
        data_vars={
            "signal": (signal_coords, signals),
            "label": (["samples"], labels),
            "segments": (["samples"], segments),
            "patient_names": (["samples"], patient_names),
            "recording_numbers": (["samples"], recording_numbers),
        },
        coords={
            "samples": range(signals.shape[0]),
        },
    )

    # show summary
    print(ds)

    # save to file
    ds.to_netcdf(args.output_path)

    # statistics
    df = ds[["label", "patient_names", "samples"]].to_dataframe()
    print(df.value_counts().sort_index())


# ===========================================================================
# Entry point
# ===========================================================================


def main(args: CliArguments):
    # create temp dir
    os.makedirs(args.temp_dir, exist_ok=True)

    # run all jobs
    epoch_data(args)
    merge(args)

    # clean up temp files
    print("CLEAN UP!")
    shutil.rmtree(args.temp_dir)

    print("All clear! File saved at:", args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset-path", type=str)  # ../data/edf/**/*.edf
    parser.add_argument("output-path", type=str)  # ../data/dataset_overlap_60.nc

    # mne
    parser.add_argument("--low-pass", action="store_true")
    parser.add_argument("--auto-reject", action="store_true")
    parser.add_argument("--ica", action="store_true")

    # epoching
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--overlap", type=int, default=0)

    # dwt
    parser.add_argument("--wavelet", type=str, choices=["mexh", "db4", "bior2.6"])
    parser.add_argument("--feat-stats", action="store_true")
    parser.add_argument("--feat-entropy", action="store_true")

    # parallel jobs
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--temp-dir", type=str, default="./tmp")
    parser.add_argument("--debug", action="store_true")

    # parse CLI
    args = CliArguments.parse(parser.parse_args())
    print(repr(args))

    # start the app!
    main(args)
