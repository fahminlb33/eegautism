import time
import glob
import json
import pathlib
import mne
import pywt

import numpy as np
from joblib import Parallel, delayed


@delayed
def derive_cwt(path: str):
    filepath = pathlib.Path(path)

    try:
        raw = mne.io.read_raw_edf(path, verbose=False)
        raw.rename_channels({"FP2": "Fp2"}, verbose=False)
        raw.set_montage("standard_1005", verbose=False)

        epochs = mne.make_fixed_length_epochs(
            raw, duration=60, preload=False, verbose=False
        )
        segments = epochs.get_data(verbose=False)

        encoded_label = "autism" if filepath.parent.name == "autism" else "normal"

        for i in range(segments.shape[0]):
            print("file:", filepath.name, "   segment:", i)

            coeff, _ = pywt.cwt(segments[i, :, :], range(1, 128), "mexh", 1)
            res = coeff.reshape(127, -1, 16)[:, :127, :]

            for j in range(16):
                np.save(f"../data/npy_no_overlap_60/{filepath.stem}-seg_{i}.npy", res)
    except Exception as e:
        print("Failed to process", filepath.name, e)


def main():
    files = [file for file in glob.glob("../data/edf/**/*.edf")]
    jobs = [derive_cwt(file) for file in files]

    start_time = time.time()
    Parallel(n_jobs=6)(jobs)
    elapsed_time = time.time() - start_time

    print("Total processing time =", elapsed_time)


if __name__ == "__main__":
    main()
