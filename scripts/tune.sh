#!/usr/bin/env bash
set -e

# vanillacnn1d
python scripts/tune.py './data/processed/cnn1d/data_10.nc' './results' --arch vanillacnn1d --test-size 0.30
python scripts/tune.py './data/processed/cnn1d/data_10.nc' './results' --arch vanillacnn1d --test-patients "mahmud,bader,mohammed"
python scripts/tune.py './data/processed/cnn1d/data_10_2.nc' './results' --arch vanillacnn1d --test-size 0.30
python scripts/tune.py './data/processed/cnn1d/data_10_2.nc' './results' --arch vanillacnn1d --test-patients "mahmud,bader,mohammed"

# eegnet
python scripts/tune.py './data/processed/cnn1d/data_10.nc' './results' --arch eegnet --test-size 0.30
python scripts/tune.py './data/processed/cnn1d/data_10.nc' './results' --arch eegnet --test-patients "mahmud,bader,mohammed"
python scripts/tune.py './data/processed/cnn1d/data_10_2.nc' './results' --arch eegnet --test-size 0.30
python scripts/tune.py './data/processed/cnn1d/data_10_2.nc' './results' --arch eegnet --test-patients "mahmud,bader,mohammed"

# deepsleepnet
python scripts/tune.py './data/processed/cnn1d/data_10.nc' './results' --arch deepsleepnet --test-size 0.30
python scripts/tune.py './data/processed/cnn1d/data_10.nc' './results' --arch deepsleepnet --test-patients "mahmud,bader,mohammed"
python scripts/tune.py './data/processed/cnn1d/data_10_2.nc' './results' --arch deepsleepnet --test-size 0.30
python scripts/tune.py './data/processed/cnn1d/data_10_2.nc' './results' --arch deepsleepnet --test-patients "mahmud,bader,mohammed"

# chrononet
python scripts/tune.py './data/processed/cnn1d/data_10.nc' './results' --arch chrononet --test-size 0.30
python scripts/tune.py './data/processed/cnn1d/data_10.nc' './results' --arch chrononet --test-patients "mahmud,bader,mohammed"
python scripts/tune.py './data/processed/cnn1d/data_10_2.nc' './results' --arch chrononet --test-size 0.30
python scripts/tune.py './data/processed/cnn1d/data_10_2.nc' './results' --arch chrononet --test-patients "mahmud,bader,mohammed"

# filipnet
python scripts/tune.py './data/processed/cnn1d/data_10.nc' './results' --arch filipnet --test-size 0.30
python scripts/tune.py './data/processed/cnn1d/data_10.nc' './results' --arch filipnet --test-patients "mahmud,bader,mohammed"
python scripts/tune.py './data/processed/cnn1d/data_10_2.nc' './results' --arch filipnet --test-size 0.30
python scripts/tune.py './data/processed/cnn1d/data_10_2.nc' './results' --arch filipnet --test-patients "mahmud,bader,mohammed"
