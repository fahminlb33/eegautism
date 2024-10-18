# Paper references

dataset KAU:

- The dataset was filtered by a band-pass filter with a passband of 0.1–60 Hz, 
- and a notch filter was used with a stopband frequency of 60 Hz. 
- All EEG signals were digitized at a sampling frequency of 256 Hz. 
- The EEG collection time ranged from 12 to 40 min for autistic patients with a total of up to 173 min. 
- For neurotypical patients, the recording varied between 5 and 27 min with a total of up to 148 min.

Daftar paper referensi

### Djemal R, AlSharabi K, Ibrahim S, Alsuwailem A. EEG-Based Computer Aided Diagnosis of Autism Spectrum Disorder Using Wavelet, Entropy, and ANN. Biomed Res Int. 2017;2017:9816591. doi: 10.1155/2017/9816591. Epub 2017 Apr 18. PMID: 28484720; PMCID: PMC5412163

Data cleaning:

1. independent component analysis (ICA) for eye-artifact removal
2. elliptic band-pass filter for filtering

Electrodes closed to eye (FP1, FP2, F7, and F8) are used as reference signals for ocular-artifacts removal. After ocular-artifact removal, the signals are then filtered using elliptic band-pass filter. Ref: https://ieeexplore.ieee.org/abstract/document/7077338/

Feature extraction:

1. Segmentation 60 seconds, with and without overlap (half segment)
2. DWT using db4 mother wavelet, 4 level of decomposition
3. statistical features (mean, standard deviation, variance, skewness, and kurtosis)
4. entropy features (log energy, threshold entropies, Renyi entropy, and Shannon entropy)

Using one-minute (60 seconds) EEG segment length, we extracted 173 segments from autistic dataset and 148 segments from normal dataset. From these 321 segments, we select randomly 32 segments for testing and the remaining for training. As 10-fold cross-validation, this process is repeated 10 times and the results are averaged.

### Alturki, F.A.; AlSharabi, K.; Abdurraqeeb, A.M.; Aljalal, M. EEG Signal Analysis for Diagnosing Neurological Disorders Using Discrete Wavelet Transform and Intelligent Techniques. Sensors 2020, 20, 2505. https://doi.org/10.3390/s20092505

Data cleaning:

1. eye artifacts have been removed from the recorded signals by ICA technique
2. elliptic band pass (0.1-60Hz) -> IIR/FIR

Feature extraction:

1. segmentasi 50 detik tanpa overlap
2. DWT
3. statistical features: LBP, SD, variance, kurtosis, and entropy

### Tawhid MNA, Siuly S, Wang H, Whittaker F, Wang K, Zhang Y. A spectrogram image based intelligent technique for automatic detection of autism spectrum disorder from EEG. PLoS One. 2021 Jun 25;16(6):e0253094. doi: 10.1371/journal.pone.0253094. PMID: 34170979; PMCID: PMC8232415.

Data cleaning:

1. Common average referencing (CAR) is used for re-referencing, in which the average value of all electrode collection (common average) is used as reference
2. In the second step of pre-processing, infinite impulse response (IIR) filter is used to low pass filter the signal at 40Hz cut off frequency
3. and finally the filtered signals from each electrode is normalized to the interval [-1, 1]. 
4. After that, pre-processed signals are segmented into 3.5 second window frames for each subject of the dataset. 

Feature extraction:

1. Using Short-Time Fourier Transform (STFT) for each of the above segments, the spectrogram plot is generated in the last step and saved as image.
