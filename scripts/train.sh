# vanillacnn1d
python train.py '../data/data_10.nc' '../data/models2' --name vanillacnn1d_10_srs30 --model vanillacnn1d --test-size 0.30
python train.py '../data/data_10.nc' '../data/models2' --name vanillacnn1d_10_ps3 --model vanillacnn1d --test-patients "mahmud,bader,mohammed"
python train.py '../data/data_10_2.nc' '../data/models2' --name vanillacnn1d_10_2_srs30 --model vanillacnn1d --test-size 0.30
python train.py '../data/data_10_2.nc' '../data/models2' --name vanillacnn1d_10_2_ps3 --model vanillacnn1d --test-patients "mahmud,bader,mohammed"

# eegnet
python train.py '../data/data_10.nc' '../data/models2' --name eegnet_10_srs30 --model eegnet --test-size 0.30
python train.py '../data/data_10.nc' '../data/models2' --name eegnet_10_ps3 --model eegnet --test-patients "mahmud,bader,mohammed"
python train.py '../data/data_10_2.nc' '../data/models2' --name eegnet_10_2_srs30 --model eegnet --test-size 0.30
python train.py '../data/data_10_2.nc' '../data/models2' --name eegnet_10_2_ps3 --model eegnet --test-patients "mahmud,bader,mohammed"

# deepsleepnet
python train.py '../data/data_10.nc' '../data/models2' --name deepsleepnet_10_srs30 --model deepsleepnet --test-size 0.30
python train.py '../data/data_10.nc' '../data/models2' --name deepsleepnet_10_ps3 --model deepsleepnet --test-patients "mahmud,bader,mohammed"
python train.py '../data/data_10_2.nc' '../data/models2' --name deepsleepnet_10_2_srs30 --model deepsleepnet --test-size 0.30
python train.py '../data/data_10_2.nc' '../data/models2' --name deepsleepnet_10_2_ps3 --model deepsleepnet --test-patients "mahmud,bader,mohammed"

# chrononet
python train.py '../data/data_10.nc' '../data/models2' --name chrononet_10_srs30 --model chrononet --test-size 0.30
python train.py '../data/data_10.nc' '../data/models2' --name chrononet_10_ps3 --model chrononet --test-patients "mahmud,bader,mohammed"
python train.py '../data/data_10_2.nc' '../data/models2' --name chrononet_10_2_srs30 --model chrononet --test-size 0.30
python train.py '../data/data_10_2.nc' '../data/models2' --name chrononet_10_2_ps3 --model chrononet --test-patients "mahmud,bader,mohammed"

# filipnet
python train.py '../data/data_10.nc' '../data/models2' --name filipnet_10_srs30 --model filipnet --test-size 0.30
python train.py '../data/data_10.nc' '../data/models2' --name filipnet_10_ps3 --model filipnet --test-patients "mahmud,bader,mohammed"
python train.py '../data/data_10_2.nc' '../data/models2' --name filipnet_10_2_srs30 --model filipnet --test-size 0.30
python train.py '../data/data_10_2.nc' '../data/models2' --name filipnet_10_2_ps3 --model filipnet --test-patients "mahmud,bader,mohammed"
