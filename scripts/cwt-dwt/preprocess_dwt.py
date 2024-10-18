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

import numpy as np
import xarray as xr
from PIL import Image
import scipy.stats
from joblib import Parallel, delayed

import matplotlib
from matplotlib.figure import Figure


@dataclasses.dataclass
class CliArguments:
    dataset_path: str
    output_path: str
    mode: str

    epoch_duration: int
    wavelet: str
    packed: bool

    feat_stats: bool
    feat_entropy: bool
    feat_signal: bool

    feat_image: bool

    temp_dir: str
    jobs: int

    @staticmethod
    def parse(args):
        norm = {k.replace("-", "_"): v for k, v in vars(args).items()}

        if args.wavelet == "db4" and (
            args.feat_entropy or args.feat_stats
        ):  # db4 => stats, entropy
            mode = "dwt-stat-entropy-packed"
        elif (
            args.wavelet == "db4" and args.feat_signal and args.packed
        ):  # db4 => signal, packed
            mode = "dwt-signal-packed"
        elif (
            args.wavelet == "db4" and args.feat_signal and not args.packed
        ):  # db4 => signal, nonpacked
            mode = "dwt-signal-unpacked"
        elif (
            args.wavelet == "mexh" and not args.feat_image and args.packed
        ):  # mexh => packed
            mode = "cwt-packed"
        elif (
            args.wavelet == "mexh" and not args.feat_image and not args.packed
        ):  # mexh => unpacked
            mode = "cwt-unpacked"
        elif (
            args.wavelet == "mexh" and args.feat_image and args.packed
        ):  # mexh => image, packed
            mode = "cwt-image-unpacked"
        else:
            raise ValueError("Unknown switch!")

        return CliArguments(mode=mode, **norm)


# ===========================================================================
# Worker jobs
# ===========================================================================


class PreprocessJob:
    def __init__(self, file_path: str, args: CliArguments) -> None:
        self.file_path = pathlib.Path(file_path)
        self.args = args

        self.segments = None

    def __call__(self):
        # read segments
        self.read_segments()

        # check if we successfully read segment, if not, return
        if self.segments is None:
            return

        # process each segment
        for i in range(self.segments.shape[0]):
            print("   file:", self.file_path.name, "   segment:", i)

            if args.wavelet == "mexh":
                self.preprocess_cwt(i)
            else:
                self.preprocess_dwt(i)

    def read_segments(self):
        try:
            # read the file
            raw = mne.io.read_raw_edf(self.file_path.absolute(), verbose=False)
            print("PROCESS!   file:", self.file_path.name)

            # update channel and montage
            raw.rename_channels({"FP2": "Fp2"}, verbose=False)
            raw.set_montage("standard_1005", verbose=False)

            # create epoch by fixed length
            epochs = mne.make_fixed_length_epochs(
                raw, duration=self.args.epoch_duration, preload=False, verbose=False
            )

            # get the segments
            self.segments = epochs.get_data(
                verbose=False
            )  # (n_epochs, n_channels, n_times)
        except Exception as e:
            print(f"FAIL!   file:{self.file_path.name}   error: {e}")

    def preprocess_cwt(self, segment_idx: int):
        # perform CWT
        coeff, _ = pywt.cwt(
            self.segments[segment_idx, :, :], range(1, 128), args.wavelet, 1
        )

        # reshape the CWT
        res = coeff.reshape(127, -1, self.segments.shape[1])[:, :127, :]

        # process each channel
        for j in range(res.shape[2]):
            img = res[:, :, j]

            # extract image
            if self.args.feat_image:
                # plot image
                fig = Figure()
                ax = fig.subplots()

                ax.imshow(res[:, :, j])
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])

                # save to memory
                buf = BytesIO()
                fig.savefig(buf, bbox_inches="tight", pad_inches=0)

                # decode
                buf.seek(0)
                img = np.array(Image.open(buf))[:, :, :3]
            else:
                # min-max
                img = (PreprocessJob.min_max(img) * 255).astype("uint8")

            # save npy
            save_path = (
                pathlib.Path(args.temp_dir)
                / f"{self.file_path.stem}-seg_{segment_idx}-c_{j}.npy"
            )
            np.save(save_path.absolute(), img)

    def preprocess_dwt(self, segment_idx: int):
        # signal output, just 1 level DWT
        if self.args.feat_signal:
            # perform DWT
            _, cD = pywt.dwt(self.segments[segment_idx, :, :], args.wavelet)

            # process each channel
            for c in range(cD.shape[0]):
                # save npy
                save_path = (
                    pathlib.Path(args.temp_dir)
                    / f"{self.file_path.stem}-seg_{segment_idx}-c_{c}.npy"
                )
                np.save(save_path.absolute(), cD[c, :])

        # 5 level DWT stats
        else:
            # perform decomposition
            decomposed = pywt.wavedec(
                self.segments[segment_idx, :, :], self.args.wavelet, level=5
            )

            # for each channel in the input data,
            for c in range(self.segments.shape[1]):
                features = []

                # derive stats/entropy data from all of the decomposition
                for decom in decomposed:
                    # extract statistical features
                    if self.args.feat_stats:
                        features += PreprocessJob.calculate_statistics(decom[c, :])

                    # extract statistical features
                    if self.args.feat_entropy:
                        features += PreprocessJob.calculate_entropy(decom[c, :])

                # save npy
                save_path = (
                    pathlib.Path(args.temp_dir)
                    / f"{self.file_path.stem}-seg_{segment_idx}-c_{c}.npy"
                )
                np.save(save_path.absolute(), np.array(features))

    @staticmethod
    def min_max(m):
        min_val = np.min(m)
        max_val = np.max(m)

        return (m - min_val) / (max_val - min_val)

    @staticmethod
    def calculate_statistics(list_values):
        return [
            np.nanmean(list_values),
            np.nanstd(list_values),
            np.nanvar(list_values),
            scipy.stats.skew(list_values),
            scipy.stats.kurtosis(list_values),
        ]

    @staticmethod
    def calculate_entropy(list_values):
        counter_values = Counter(list_values).most_common()
        probabilities = [elem[1] / len(list_values) for elem in counter_values]

        threshold_entropy = 0.2
        renyi_alpha = 2

        return [
            # log energy
            np.sum(np.log(np.square(probabilities))),
            # threshold entropy
            np.sum(np.where(np.array(probabilities) > threshold_entropy, 1, 0)),
            # renyi entropy
            np.log(np.sum(np.power(probabilities, renyi_alpha))) * (1 / renyi_alpha),
            # shannon
            scipy.stats.entropy(probabilities),
        ]


# ===========================================================================
# Pipeline jobs
# ===========================================================================


def derive(args: CliArguments):
    print("JOB!   Deriving wavelet...")

    # find all EDF files
    files = [file for file in glob.glob(args.dataset_path)]

    # derive using wavelet
    jobs = [delayed(lambda x: PreprocessJob(x, args)())(file) for file in files]

    # execute jobs in parallel
    start_time = time.time()
    Parallel(n_jobs=args.jobs)(jobs)
    elapsed_time = time.time() - start_time

    print("Total processing time =", elapsed_time)


def merge(args: CliArguments):
    print("JOB!   Merging features...")

    # find all npy files
    files = {}
    search_path = pathlib.Path(args.temp_dir) / "*.npy"
    for file in glob.glob(str(search_path.absolute())):
        # get filename
        filename = pathlib.Path(file)
        name_no_channel = re.sub("-c_\d+", "", filename.stem)

        if "packed" not in args.mode:
            # separate operation
            files[filename.stem] = [file]
            continue

        # check if the array is empty
        if name_no_channel not in files:
            files[name_no_channel] = []

        # append to packer
        files[name_no_channel].append(file)

    # to store the output
    X = []
    y = []

    # process all files
    for root_name, file_channels in files.items():
        if args.mode == "dwt-stat-entropy-packed":
            X.append(np.concatenate([np.load(file).ravel() for file in file_channels]))
            y.append(1 if "Autism" in root_name else 0)
        elif args.mode == "dwt-signal-packed" or args.mode == "cwt-packed":
            X.append(np.array([np.load(file) for file in file_channels]))
            y.append(1 if "Autism" in root_name else 0)
        else:
            X.extend([np.load(file) for file in file_channels])
            y.extend([1 if "Autism" in root_name else 0] * len(file_channels))

    # cast to array
    X = np.array(X)
    y = np.array(y)

    # reshape output tensor
    coords = []
    if args.mode == "dwt-stat-entropy-packed":
        coords = ["samples", "features"]
    elif args.mode == "dwt-signal-packed" or args.mode == "dwt-signal-unpacked":
        coords = ["samples", "time_step", "channels"]
        if args.mode == "dwt-signal-packed":
            X = np.moveaxis(X, 1, -1)
            pass
        elif args.mode == "dwt-signal-unpacked":
            X = np.expand_dims(X, axis=-1)
    elif (
        args.mode == "cwt-packed"
        or args.mode == "cwt-unpacked"
        or args.mode == "cwt-image-unpacked"
    ):
        coords = ["samples", "frequency", "time", "channels"]
        if args.mode == "cwt-unpacked":
            X = np.expand_dims(X, axis=-1)
        elif args.mode == "cwt-packed":
            X = np.moveaxis(X, 1, -1)

    print(X.shape)
    print(y.shape)

    # combine all
    ds = xr.Dataset(
        data_vars={
            "features": (coords, X),
            "labels": (["samples"], y),
        },
        coords={"samples": range(y.shape[0])},
    )

    # show summary
    print(ds)

    # save to file
    ds.to_netcdf(args.output_path)


# ===========================================================================
# Entry point
# ===========================================================================


def main(args: CliArguments):
    # create temp dir
    os.makedirs(args.temp_dir, exist_ok=True)

    # run all jobs
    derive(args)
    merge(args)

    # clean up temp files
    print("CLEAN UP!")
    # shutil.rmtree(args.temp_dir)

    print("All clear! File saved at:", args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset-path", type=str)  # ../data/edf/**/*.edf
    parser.add_argument("output-path", type=str)  # ../data/dataset_overlap_60.nc

    # wavelet config
    parser.add_argument("--epoch-duration", type=int, default=60)
    parser.add_argument(
        "--wavelet", type=str, choices=["mexh", "db4", "bior2.6"], default="mexh"
    )
    parser.add_argument("--packed", action="store_true")

    # DWT specific
    parser.add_argument("--feat-stats", action="store_true")
    parser.add_argument("--feat-entropy", action="store_true")
    parser.add_argument("--feat-signal", action="store_true")

    # CWT specific
    parser.add_argument("--feat-image", action="store_true")

    # parallel jobs
    parser.add_argument("--temp-dir", type=str, default="./tmp")
    parser.add_argument("--jobs", type=int, default=4)

    # parse CLI
    args = CliArguments.parse(parser.parse_args())
    pprint(repr(args))

    # use Agg backend
    matplotlib.use("Agg")

    # start the app!
    main(args)
