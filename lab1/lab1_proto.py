# DT2119, Lab 1 Feature Extraction
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.signal.windows import hamming
from scipy.fftpack import fft, dct
from lab1_tools import trfbank, lifter, tidigit2labels
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage

data = np.load('lab1_data.npz', allow_pickle=True)['data']
example = np.load('lab1_example.npz', allow_pickle=True)['example'].item()

aa=10
# Function given by the exercise ----------------------------------

def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    return lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    num_windows = 1 + (len(samples) - winlen) // winshift
    frames = np.zeros((num_windows, winlen))
    
    for i in range(num_windows):
        start = i * winshift
        end = start + winlen
        frames[i, :] = samples[start:end]
    
    return frames
    
def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    
    b = np.array([1, -p])  
    a = np.array([1])    

    preemph = lfilter(b, a, input)
    return preemph

def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """

    M = input.shape[1]
    window = hamming(M, sym=False)
    windowed = input * window
    
    return windowed

def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """

    fft_result = fft(input, n=nfft)
    power_spec = np.abs(fft_result) ** 2
    
    return power_spec

def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """

    nfft = input.shape[1]
    mel_filterbank = trfbank(samplingrate, nfft)
    mel_spectrum = np.dot(input, mel_filterbank.T) 
    log_mel_spectrum = np.log(mel_spectrum) 
    
    # freqs = np.linspace(0, samplingrate / 2, nfft // 2 + 1)
    # plt.figure(figsize=(10, 6))
    
    # for i in range(mel_filterbank.shape[0]):
    #     plt.plot(freqs, mel_filterbank[i, :nfft // 2 + 1], label=f'Filter {i+1}')

    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Amplitude')
    # plt.title('Mel Filterbank in Linear Frequency Scale')
    # plt.show()
    return log_mel_spectrum

def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """

    cepstral_coeffs = dct(input, type=2, axis=1, n=nceps)
    return cepstral_coeffs

def plt_draw(frame):
    plt.figure(figsize=(10, 2))
    plt.pcolormesh(frame.T)
    plt.colorbar()
    plt.show()

def dist_func(x, y):
    return np.linalg.norm(x - y)

def dtw(x, y, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """

    N = x.shape[0]
    M = y.shape[0]
    LD = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            LD[i, j] = dist(x[i], y[j])
    
    # initialize accumulated distance matrix
    AD = np.inf * np.ones((N, M))
    AD[0, 0] = LD[0, 0]
    for i in range(1, N):
        AD[i, 0] = AD[i-1, 0] + LD[i, 0]
    for j in range(1, M):
        AD[0, j] = AD[0, j-1] + LD[0, j]
    for i in range(1, N):
        for j in range(1, M):
            AD[i, j] = LD[i, j] + min(AD[i-1, j-1], AD[i-1,j], AD[i, j-1])
    d = AD[N-1][M-1] / (N + M)
    return d, LD, AD

def calc_feature_correlation(feature, plot=False):
    correlation_matrix = np.corrcoef(feature.T)

    if plot:
        plt.figure(figsize=(10, 8))
        
        num_features = correlation_matrix.shape[0]
        x = np.arange(num_features + 1)
        y = np.arange(num_features + 1)
        
        plt.pcolormesh(x, y, correlation_matrix, cmap='coolwarm', shading='auto')
        plt.colorbar()

        plt.title('Feature Correlation Matrix')
        plt.xlabel('Features')
        plt.ylabel('Features')

        plt.xticks(np.arange(num_features) + 0.5, range(num_features))
        plt.yticks(np.arange(num_features) + 0.5, range(num_features))
        
        plt.xlim(0, num_features)
        plt.ylim(0, num_features)
        
        plt.show()
    
    return correlation_matrix

def gmm_cluster(feature, num_comp=4):
    gmm = GaussianMixture(n_components=num_comp, random_state=0, verbose=1).fit(feature)
    return gmm

def gmm_posterior(gmm, test_feature_list):
    post_res = []
    for test_feature in test_feature_list:
        posteriors = gmm.predict_proba(test_feature)
        post_res.append(posteriors)
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))

    cax1 = axs[0, 0].pcolormesh(post_res[0].T, shading='auto')
    axs[0, 0].set_ylabel('GMM component')
    axs[0, 0].set_xlabel('Time (frame index)')

    cax2 = axs[0, 1].pcolormesh(post_res[1].T, shading='auto')
    axs[0, 1].set_ylabel('GMM component')
    axs[0, 1].set_xlabel('Time (frame index)')

    cax3 = axs[1, 0].pcolormesh(post_res[2].T, shading='auto')
    axs[1, 0].set_ylabel('GMM component')
    axs[1, 0].set_xlabel('Time (frame index)')

    cax4 = axs[1, 1].pcolormesh(post_res[3].T, shading='auto')
    axs[1, 1].set_ylabel('GMM component')
    axs[1, 1].set_xlabel('Time (frame index)')

    fig.colorbar(cax4, ax=axs.ravel().tolist(), label='Posterior probability')
    plt.show()


def main():
# 4. Mel Frequency Cepstrum Coefficients step-by-step

    winlen = 400
    winshift = 200
    nfft = 512
    nceps = 13

    eg_frames = enframe(example['samples'],winlen,winshift)
    eg_preemph = preemp(eg_frames)
    eg_windowed = windowing(eg_preemph)
    eg_spec = powerSpectrum(eg_windowed, nfft)
    eg_mspec = mspec(example['samples'])
    eg_mfcc = cepstrum(eg_mspec, nceps)
    eg_lmfcc = mfcc(example['samples'])

    plt_draw(eg_frames)
    plt_draw(eg_preemph)
    plt_draw(eg_windowed)
    plt_draw(eg_spec)
    plt_draw(eg_mspec)
    plt_draw(eg_mfcc)
    plt_draw(eg_lmfcc)


    # for utterance in data:
    #     mfcc_res = mfcc(utterance['samples'], samplingrate=utterance['samplingrate'])
    #     plt_draw(mfcc_res)

    aa=10 

# 5. Feature Correlation

    feature = None
    for utterance in data:
        mfcc_res = mfcc(utterance['samples'], samplingrate=utterance['samplingrate'])
        feature = mfcc_res if feature is None else np.concatenate((feature, mfcc_res), axis=0)

    calc_feature_correlation(feature, plot=True)

    feature_mspec = None
    for utterance in data:
        mspec_res = mspec(utterance['samples'], samplingrate=utterance['samplingrate'])
        feature_mspec = mspec_res if feature_mspec is None else np.concatenate((feature_mspec, mspec_res), axis=0)

    calc_feature_correlation(feature_mspec, plot=True)


# 6. Speech Segments with Clustering

    gmm = gmm_cluster(feature, num_comp=32)
    test_feature_list = []
    for utterance in data[[16, 17, 38, 39]]: 
        mfcc_res = mfcc(utterance['samples'], samplingrate=utterance['samplingrate'])
        test_feature_list.append(mfcc_res)
    gmm_posterior(gmm, test_feature_list)


# 7. Comparing Utterances

    utterance = data[0]
    mfcc_x = mfcc(utterance['samples'], samplingrate=utterance['samplingrate'])
    utterance = data[1]
    mfcc_y = mfcc(utterance['samples'], samplingrate=utterance['samplingrate'])
    dtw(mfcc_x, mfcc_y, dist=dist_func)

    D = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
            sub_data = data[i]
            mfcc_x = mfcc(sub_data['samples'], samplingrate=sub_data['samplingrate'])
            sub_data = data[j]
            mfcc_y = mfcc(sub_data['samples'], samplingrate=sub_data['samplingrate'])
            D[i, j], _, _ = dtw(mfcc_x, mfcc_y, dist=dist_func)
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(D, shading='auto')
    plt.colorbar(label='Distance')
    plt.title('global pairwise distances')
    plt.show()

    D = np.load('D_res.npy')
    Z = linkage(D, 'complete')
    plt.figure(figsize=(10, 8))
    dendrogram(Z, labels=tidigit2labels(data), leaf_rotation=90)
    plt.title('dendrogram')
    plt.show()

    aa = 10


if __name__ == '__main__':
    main()