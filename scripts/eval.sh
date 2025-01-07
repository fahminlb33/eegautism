#!/usr/bin/env bash
set -e

# vanillacnn1d
python scripts/evaluate.py './data/processed/cnn1d/data_10.nc' './results' \
    --name vanillacnn1d_10_srs30 \
    --arch vanillacnn1d \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 0.001 \
    --test-size 0.30
python scripts/evaluate.py './data/processed/cnn1d/data_10.nc' './results' \
    --name vanillacnn1d_10_ps3 \
    --arch vanillacnn1d \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.01 \
    --test-patients "mahmud,bader,mohammed"
python scripts/evaluate.py './data/processed/cnn1d/data_10_2.nc' './results' \
    --name vanillacnn1d_10_2_srs30 \
    --arch vanillacnn1d \
    --epochs 50 \
    --batch-size 16 \
    --learning-rate 0.01 \
    --test-size 0.30
python scripts/evaluate.py './data/processed/cnn1d/data_10_2.nc' './results' \
    --name vanillacnn1d_10_2_ps3 \
    --arch vanillacnn1d \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.01 \
    --test-patients "mahmud,bader,mohammed"

# eegnet
python scripts/evaluate.py './data/processed/cnn1d/data_10.nc' './results' \
    --name eegnet_10_srs30 \
    --arch eegnet \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 0.01 \
    --test-size 0.30
python scripts/evaluate.py './data/processed/cnn1d/data_10.nc' './results' \
    --name eegnet_10_ps3 \
    --arch eegnet \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 0.001 \
    --test-patients "mahmud,bader,mohammed"
python scripts/evaluate.py './data/processed/cnn1d/data_10_2.nc' './results' \
    --name eegnet_10_2_srs30 \
    --arch eegnet \
    --epochs 100 \
    --batch-size 8 \
    --learning-rate 0.001 \
    --test-size 0.30
python scripts/evaluate.py './data/processed/cnn1d/data_10_2.nc' './results' \
    --name eegnet_10_2_ps3 \
    --arch eegnet \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --test-patients "mahmud,bader,mohammed"

# deepsleepnet
python scripts/evaluate.py './data/processed/cnn1d/data_10.nc' './results' \
    --name deepsleepnet_10_srs30 \
    --arch deepsleepnet \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.01 \
    --test-size 0.30
python scripts/evaluate.py './data/processed/cnn1d/data_10.nc' './results' \
    --name deepsleepnet_10_ps3 \
    --arch deepsleepnet \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --test-patients "mahmud,bader,mohammed"
python scripts/evaluate.py './data/processed/cnn1d/data_10_2.nc' './results' \
    --name deepsleepnet_10_2_srs30 \
    --arch deepsleepnet \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 0.01 \
    --test-size 0.30
python scripts/evaluate.py './data/processed/cnn1d/data_10_2.nc' './results' \
    --name deepsleepnet_10_2_ps3 \
    --arch deepsleepnet \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --test-patients "mahmud,bader,mohammed"

# chrononet
python scripts/evaluate.py './data/processed/cnn1d/data_10.nc' './results' \
    --name chrononet_10_srs30 \
    --arch chrononet \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --test-size 0.30
python scripts/evaluate.py './data/processed/cnn1d/data_10.nc' './results' \
    --name chrononet_10_ps3 \
    --arch chrononet \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --test-patients "mahmud,bader,mohammed"
python scripts/evaluate.py './data/processed/cnn1d/data_10_2.nc' './results' \
    --name chrononet_10_2_srs30 \
    --arch chrononet \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --test-size 0.30
python scripts/evaluate.py './data/processed/cnn1d/data_10_2.nc' './results' \
    --name chrononet_10_2_ps3 \
    --arch chrononet \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.01 \
    --test-patients "mahmud,bader,mohammed"

# filipnet
python scripts/evaluate.py './data/processed/cnn1d/data_10.nc' './results' \
    --name filipnet_10_srs30 \
    --arch filipnet \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.01 \
    --test-size 0.30
python scripts/evaluate.py './data/processed/cnn1d/data_10.nc' './results' \
    --name filipnet_10_ps3 \
    --arch filipnet \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --test-patients "mahmud,bader,mohammed"
python scripts/evaluate.py './data/processed/cnn1d/data_10_2.nc' './results' \
    --name filipnet_10_2_srs30 \
    --arch filipnet \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --test-size 0.30
python scripts/evaluate.py './data/processed/cnn1d/data_10_2.nc' './results' \
    --name filipnet_10_2_ps3 \
    --arch filipnet \
    --epochs 25 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --test-patients "mahmud,bader,mohammed"
