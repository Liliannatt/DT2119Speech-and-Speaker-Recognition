import numpy as np
from scipy.special import logsumexp
from lab2_tools import *
import matplotlib.pyplot as plt


#5.1 Emission probabilities
def concatTwoHMMs(hmm1, hmm2):
    """ Concatenates 2 HMM models

    Args:
       hmm1, hmm2: two dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be different for each)

    Output
       dictionary with the same keys as the input but concatenated models:
          startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models
   
    Example:
       twoHMMs = concatHMMs(phoneHMMs['sil'], phoneHMMs['ow'])

    See also: the concatenating_hmms.pdf document in the lab package
    """
    M1 = len(hmm1['startprob']) - 1  
    M2 = len(hmm2['startprob']) - 1
    concatenated_hmm = {
        'name': hmm1['name'] + '+' + hmm2['name'],
        'startprob': np.hstack([hmm1['startprob'][:-1], hmm2['startprob']]),
        'transmat': np.block([
            [hmm1['transmat'][:-1, :-1], np.zeros((M1, M2+1))],
            [np.zeros((M2+1, M1)), hmm2['transmat']]
        ]),
        'means': np.vstack([hmm1['means'], hmm2['means']]),
        'covars': np.vstack([hmm1['covars'], hmm2['covars']])
    }

    concatenated_hmm['startprob'][:M1] = hmm1['startprob'][:-1]
    concatenated_hmm['startprob'][M1:] = hmm1['startprob'][-1] * hmm2['startprob']
    concatenated_hmm['transmat'][:M1, :M1] = hmm1['transmat'][:-1, :-1]
    concatenated_hmm['transmat'][:M1, M1:] = hmm1['transmat'][:M1, -1].reshape(-1, 1) * hmm2['startprob'].reshape(1, -1)
    concatenated_hmm['transmat'][M1:, M1:] = hmm2['transmat']

    return concatenated_hmm



# this is already implemented, but based on concat2HMMs() above
def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: dictionary of models indexed by model name. 
       hmmmodels[name] is a dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models:
         startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    concat = hmmmodels[namelist[0]]
    for idx in range(1,len(namelist)):
        concat = concatTwoHMMs(concat, hmmmodels[namelist[idx]])
    return concat

#obsloglik: HMM log likelihood of observation given the state
prondict = {
    'o': ['ow'],
    'z': ['z', 'iy', 'r', 'ow'],
    '1': ['w', 'ah', 'n'],
    '2': ['t', 'uw'],
    '3': ['th', 'r', 'iy'],
    '4': ['f', 'ao', 'r'],
    '5': ['f', 'ay', 'v'],
    '6': ['s', 'ih', 'k', 's'],
    '7': ['s', 'eh', 'v', 'ah', 'n'],
    '8': ['ey', 't'],
    '9': ['n', 'ay', 'n'],
}

#data
data = np.load('lab2_data.npz', allow_pickle=True)['data']
example = np.load('lab2_example.npz', allow_pickle=True)['example'].item()

# phonetic models (HMMs for phonemes)
phoneHMMs_onespkr = np.load('lab2_models_onespkr.npz', allow_pickle=True)['phoneHMMs'].item()
phoneHMMs_all = np.load('lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()

# lexical model (isolated word recognition => start and end silence)
isolated = {}
for digit in prondict.keys():
    isolated[digit] = ['sil'] + prondict[digit] + ['sil']

# word models (HMMs for words)
wordHMMs_onespkr = {}
wordHMMs_all = {}
for digit in prondict.keys():
    wordHMMs_onespkr[digit] = concatHMMs(phoneHMMs_onespkr, isolated[digit])
    wordHMMs_onespkr[digit]['name'] = digit

    wordHMMs_all[digit] = concatHMMs(phoneHMMs_all, isolated[digit])
    wordHMMs_all[digit]['name'] = digit

wordHMM = wordHMMs_onespkr[example['digit']]
obsloglik = log_multivariate_normal_density_diag(example['lmfcc'], wordHMM['means'], wordHMM['covars'])

plt.figure()
pcolormesh(obsloglik, 'Emission probabilities - computed')






def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    weighted_log_emlik = weights * log_emlik
    return np.sum(weighted_log_emlik)


#5.2 forward algorithm
def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """
    log_alpha = np.zeros(log_emlik.shape)
    log_startprob = log_startprob[:-1]     
    log_transmat = log_transmat[:-1, :-1]

    log_alpha[0] = log_startprob + log_emlik[0]
    for n in range(1, len(log_emlik)):
        log_alpha[n] = logsumexp(log_alpha[n-1].reshape(-1, 1) + log_transmat) + log_emlik[n]

    loglik = logsumexp(log_alpha[-1])
    return log_alpha, loglik


logalpha, loglik = forward(obsloglik, np.log(wordHMM['startprob']), np.log(wordHMM['transmat']))
plt.figure()
pcolormesh(logalpha, r'Forward log probablities')



#5.4 backward algorithm
def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """
    log_beta = np.zeros(log_emlik.shape)
    log_startprob = log_startprob[:-1]
    log_transmat = log_transmat[:-1, :-1]

    log_beta[-1] = 0
    for n in range(len(log_emlik)-2, -1, -1):
        log_beta[n] = logsumexp(log_transmat + log_emlik[n+1] + log_beta[n+1], axis=1)

    loglik = logsumexp(log_emlik[0] + log_beta[0] + log_startprob)
    return log_beta, loglik

logbeta, loglik = backward(obsloglik, np.log(wordHMM['startprob']), np.log(wordHMM['transmat']))

plt.figure()
pcolormesh(logbeta, r'Backward log probablities')



#5.3 viterbi approximation
def viterbi(log_emlik, log_startprob, log_transmat, forceFinalState=True):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        forceFinalState: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    N, M = log_emlik.shape
    logViterbi = np.zeros((N, M))
    B = np.zeros((N, M), dtype='uint32')    
    log_startprob = log_startprob[:-1]      # remove non-emitting state
    log_transmat = log_transmat[:-1, :-1]

    logViterbi[0] = log_startprob + log_emlik[0]
    for n in range(1, len(log_emlik)):
        aux = logViterbi[n - 1].reshape(-1, 1) + log_transmat
        logViterbi[n] = np.max(aux, axis=0) + log_emlik[n]
        B[n] = np.argmax(aux, axis=0)
    viterbi_loglik = np.max(logViterbi[-1])

    # backtracking
    viterbi_path = np.zeros(N, dtype='uint32')
    if forceFinalState:
        viterbi_path[-1] = M-1
    else:
        viterbi_path[-1] = np.argmax(logViterbi[-1])
    for n in range(N-2, -1, -1):
        viterbi_path[n] = B[n+1, viterbi_path[n+1]]

    return viterbi_loglik, viterbi_path

vloglik, vpath = viterbi(obsloglik, np.log(wordHMM['startprob']), np.log(wordHMM['transmat']))

# alphas overlaid by best best
plt.figure()
pcolormesh(logalpha, r'best path')
plt.plot(np.arange(len(vpath)) + .5, vpath + .5, 'r-', linewidth=1, label='computed')
plt.legend()



#6.1 state posteriors
def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """
    
    log_gamma = log_alpha + log_beta - logsumexp(log_alpha[-1])
    return log_gamma

loggamma = statePosteriors(logalpha, logbeta)

plt.figure()
pcolormesh(loggamma, r'Log $\gamma$ - computed')

# row-stochasticx
if not all(np.isclose(np.exp(logsumexp(loggamma, axis=1)), 1)):
    raise ValueError('log_gamma is not row-stochastic.')

# GMMs vs HMMs
loggamma_gmm = obsloglik - logsumexp(obsloglik, axis=1).reshape(-1, 1)
plt.figure()
pcolormesh(loggamma_gmm, r'Log $\gamma^{GMM}$')



#6.2 retraining emmision distributions
def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """
    N, D = X.shape
    M = log_gamma.shape[1]
    means = np.zeros((M, D))
    covars = np.zeros((M, D))

    for j in range(M):
        means[j] = np.sum(np.exp(log_gamma[:, j]) * X.T, axis=1) / np.exp(logsumexp(log_gamma[:, j]))
        covars[j] = np.sum(np.exp(log_gamma[:, j]) * (X - means[j]).T ** 2, axis=1) / np.exp(logsumexp(log_gamma[:, j]))
        covars[covars < varianceFloor] = varianceFloor
    return means, covars

