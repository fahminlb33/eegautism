import os
import re
import time
import glob
import shutil
import pathlib
import argparse
import dataclasses

import mne
import numpy as np
import xarray as xr
from autoreject import AutoReject
from joblib import Parallel, delayed


@dataclasses.dataclass
class CliArguments:
    dataset_path: str
    output_path: str

    overlap: int
    duration: int

    jobs: int
    temp_dir: str

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

        self.segments = None

    def __call__(self):
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
            filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)

            # create epoch by fixed length
            epochs = mne.make_fixed_length_epochs(
                filt_raw,
                duration=self.args.duration,
                overlap=self.args.overlap,
                preload=True,
                verbose=False,
            )

            # perform autoreject
            ar = AutoReject(
                n_interpolate=[1, 2, 3, 4], random_state=11, n_jobs=1, verbose=False
            )
            epochs, reject_log = ar.fit_transform(epochs, return_log=True)

            # perform ICA
            ica = mne.preprocessing.ICA(max_iter="auto", random_state=97, verbose=False)
            ica.fit(epochs, verbose=False)
            ica.exclude = [0, 2]
            ica.apply(epochs, exclude=ica.exclude, verbose=False)

            print(
                "Explained variance by all ICA components: ",
                ica.get_explained_variance_ratio(filt_raw)["eeg"],
            )

            # get the segments
            segments = epochs.get_data(verbose=False, copy=True)

            # save npy
            # (n_segments, n_channels, n_time_step)
            save_path = pathlib.Path(args.temp_dir) / f"{self.file_path.stem}.npy"
            np.save(save_path.absolute(), segments)

        except Exception as e:
            print(f"FAIL!   file:{self.file_path.name}   error: {e}")


# ===========================================================================
# Pipeline jobs
# ===========================================================================


def epoch_data(args: CliArguments):
    print("JOB!   Reading EEG data...")

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

    labels = []
    signals = []
    segments = []
    patient_names = []
    recording_numbers = []

    # find all npy files
    search_path = pathlib.Path(args.temp_dir) / "*.npy"
    for file in glob.glob(str(search_path.absolute())):
        # skip rejection files
        if "rejection" in file:
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

    # reshape output tensor
    signals = np.moveaxis(signals, 1, 2)

    # combine all
    ds = xr.Dataset(
        data_vars={
            "signal": (["samples", "time_steps", "channels"], signals),
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

    # wavelet config
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--overlap", type=int, default=0)

    # parallel jobs
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--temp-dir", type=str, default="./tmp")

    # parse CLI
    args = CliArguments.parse(parser.parse_args())
    print(repr(args))

    # start the app!
    main(args)
