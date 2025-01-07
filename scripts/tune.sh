#!/usr/bin/env bash
set -e

# vanillacnn1d
python scripts/tune.py './data/processed/cnn1d/data_10.nc' './results/tune-new.jsonl' --arch vanillacnn1d --test-size 0.30
python scripts/tune.py './data/processed/cnn1d/data_10.nc' './results/tune-new.jsonl' --arch vanillacnn1d --test-patients "mahmud,bader,mohammed"
python scripts/tune.py './data/processed/cnn1d/data_10_2.nc' './results/tune-new.jsonl' --arch vanillacnn1d --test-size 0.30
python scripts/tune.py './data/processed/cnn1d/data_10_2.nc' './results/tune-new.jsonl' --arch vanillacnn1d --test-patients "mahmud,bader,mohammed"

# eegnet
python scripts/tune.py './data/processed/cnn1d/data_10.nc' './results/tune-new.jsonl' --arch eegnet --test-size 0.30
python scripts/tune.py './data/processed/cnn1d/data_10.nc' './results/tune-new.jsonl' --arch eegnet --test-patients "mahmud,bader,mohammed"
python scripts/tune.py './data/processed/cnn1d/data_10_2.nc' './results/tune-new.jsonl' --arch eegnet --test-size 0.30
python scripts/tune.py './data/processed/cnn1d/data_10_2.nc' './results/tune-new.jsonl' --arch eegnet --test-patients "mahmud,bader,mohammed"

# deepsleepnet
python scripts/tune.py './data/processed/cnn1d/data_10.nc' './results/tune-new.jsonl' --arch deepsleepnet --test-size 0.30
python scripts/tune.py './data/processed/cnn1d/data_10.nc' './results/tune-new.jsonl' --arch deepsleepnet --test-patients "mahmud,bader,mohammed"
python scripts/tune.py './data/processed/cnn1d/data_10_2.nc' './results/tune-new.jsonl' --arch deepsleepnet --test-size 0.30
python scripts/tune.py './data/processed/cnn1d/data_10_2.nc' './results/tune-new.jsonl' --arch deepsleepnet --test-patients "mahmud,bader,mohammed"

# chrononet
python scripts/tune.py './data/processed/cnn1d/data_10.nc' './results/tune-new.jsonl' --arch chrononet --test-size 0.30
python scripts/tune.py './data/processed/cnn1d/data_10.nc' './results/tune-new.jsonl' --arch chrononet --test-patients "mahmud,bader,mohammed"
python scripts/tune.py './data/processed/cnn1d/data_10_2.nc' './results/tune-new.jsonl' --arch chrononet --test-size 0.30
python scripts/tune.py './data/processed/cnn1d/data_10_2.nc' './results/tune-new.jsonl' --arch chrononet --test-patients "mahmud,bader,mohammed"

# filipnet
python scripts/tune.py './data/processed/cnn1d/data_10.nc' './results/tune-new.jsonl' --arch filipnet --test-size 0.30
python scripts/tune.py './data/processed/cnn1d/data_10.nc' './results/tune-new.jsonl' --arch filipnet --test-patients "mahmud,bader,mohammed"
python scripts/tune.py './data/processed/cnn1d/data_10_2.nc' './results/tune-new.jsonl' --arch filipnet --test-size 0.30
python scripts/tune.py './data/processed/cnn1d/data_10_2.nc' './results/tune-new.jsonl' --arch filipnet --test-patients "mahmud,bader,mohammed"

sleep 5s

shutdown.exe -s -t 0
