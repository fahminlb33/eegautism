#!/usr/bin/env bash
set -e

# vanillacnn1d
python scripts/train-cv.py './data/processed/cnn1d/data_10.nc' './results/train-cv.jsonl' --name vanillacnn1d_10_srs30 --arch vanillacnn1d --test-size 0.30
python scripts/train-cv.py './data/processed/cnn1d/data_10.nc' './results/train-cv.jsonl' --name vanillacnn1d_10_ps3 --arch vanillacnn1d --test-patients "mahmud,bader,mohammed"
python scripts/train-cv.py './data/processed/cnn1d/data_10_2.nc' './results/train-cv.jsonl' --name vanillacnn1d_10_2_srs30 --arch vanillacnn1d --test-size 0.30
python scripts/train-cv.py './data/processed/cnn1d/data_10_2.nc' './results/train-cv.jsonl' --name vanillacnn1d_10_2_ps3 --arch vanillacnn1d --test-patients "mahmud,bader,mohammed"

# eegnet
python scripts/train-cv.py './data/processed/cnn1d/data_10.nc' './results/train-cv.jsonl' --name eegnet_10_srs30 --arch eegnet --test-size 0.30
python scripts/train-cv.py './data/processed/cnn1d/data_10.nc' './results/train-cv.jsonl' --name eegnet_10_ps3 --arch eegnet --test-patients "mahmud,bader,mohammed"
python scripts/train-cv.py './data/processed/cnn1d/data_10_2.nc' './results/train-cv.jsonl' --name eegnet_10_2_srs30 --arch eegnet --test-size 0.30
python scripts/train-cv.py './data/processed/cnn1d/data_10_2.nc' './results/train-cv.jsonl' --name eegnet_10_2_ps3 --arch eegnet --test-patients "mahmud,bader,mohammed"

# deepsleepnet
python scripts/train-cv.py './data/processed/cnn1d/data_10.nc' './results/train-cv.jsonl' --name deepsleepnet_10_srs30 --arch deepsleepnet --test-size 0.30
python scripts/train-cv.py './data/processed/cnn1d/data_10.nc' './results/train-cv.jsonl' --name deepsleepnet_10_ps3 --arch deepsleepnet --test-patients "mahmud,bader,mohammed"
python scripts/train-cv.py './data/processed/cnn1d/data_10_2.nc' './results/train-cv.jsonl' --name deepsleepnet_10_2_srs30 --arch deepsleepnet --test-size 0.30
python scripts/train-cv.py './data/processed/cnn1d/data_10_2.nc' './results/train-cv.jsonl' --name deepsleepnet_10_2_ps3 --arch deepsleepnet --test-patients "mahmud,bader,mohammed"

# chrononet
python scripts/train-cv.py './data/processed/cnn1d/data_10.nc' './results/train-cv.jsonl' --name chrononet_10_srs30 --arch chrononet --test-size 0.30
python scripts/train-cv.py './data/processed/cnn1d/data_10.nc' './results/train-cv.jsonl' --name chrononet_10_ps3 --arch chrononet --test-patients "mahmud,bader,mohammed"
python scripts/train-cv.py './data/processed/cnn1d/data_10_2.nc' './results/train-cv.jsonl' --name chrononet_10_2_srs30 --arch chrononet --test-size 0.30
python scripts/train-cv.py './data/processed/cnn1d/data_10_2.nc' './results/train-cv.jsonl' --name chrononet_10_2_ps3 --arch chrononet --test-patients "mahmud,bader,mohammed"

# filipnet
python scripts/train-cv.py './data/processed/cnn1d/data_10.nc' './results/train-cv.jsonl' --name filipnet_10_srs30 --arch filipnet --test-size 0.30
python scripts/train-cv.py './data/processed/cnn1d/data_10.nc' './results/train-cv.jsonl' --name filipnet_10_ps3 --arch filipnet --test-patients "mahmud,bader,mohammed"
python scripts/train-cv.py './data/processed/cnn1d/data_10_2.nc' './results/train-cv.jsonl' --name filipnet_10_2_srs30 --arch filipnet --test-size 0.30
python scripts/train-cv.py './data/processed/cnn1d/data_10_2.nc' './results/train-cv.jsonl' --name filipnet_10_2_ps3 --arch filipnet --test-patients "mahmud,bader,mohammed"
