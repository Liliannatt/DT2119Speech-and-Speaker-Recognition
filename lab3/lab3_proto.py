import numpy as np
from lab3_tools import * 
from lab1_proto import *
from prondict import prondict
from lab2_proto import *
import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

import torch
import torch.nn.functional as F

def words2phones(wordList, pronDict, addSilence=True, addShortPause=True):
    """ word2phones: converts word level to phone level transcription adding silence

    Args:
       wordList: list of word symbols
       pronDict: pronunciation dictionary. The keys correspond to words in wordList
       addSilence: if True, add initial and final silence
       addShortPause: if True, add short pause model "sp" at end of each word
    Output:
       list of phone symbols
    """
    phoneTrans = []

    if addSilence:
      phoneTrans.append('sil')

    for word in wordList:
      if word in pronDict:
         phoneTrans.extend(pronDict[word])

      if addShortPause:
         phoneTrans.append('sp')

    if addSilence:
      phoneTrans.append('sil')

    return phoneTrans

def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
    """ forcedAlignmen: aligns a phonetic transcription at the state level

    Args:
       lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
              computed the same way as for the training of phoneHMMs
       phoneHMMs: set of phonetic Gaussian HMM models
       phoneTrans: list of phonetic symbols to be aligned including initial and
                   final silence

    Returns:
       list of strings in the form phoneme_index specifying, for each time step
       the state from phoneHMMs corresponding to the viterbi path.
    """
    utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)
    nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phoneTrans}
    stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans
                  for stateid in range(nstates[phone])]

    obsloglik = log_multivariate_normal_density_diag(lmfcc, utteranceHMM['means'], utteranceHMM['covars'])

    _, vpath = viterbi(obsloglik, np.log(utteranceHMM['startprob']), np.log(utteranceHMM['transmat']))

    viterbiStateTrans = [stateTrans[state] for state in vpath]

    return viterbiStateTrans

def stack_features(features, n, context=3):
    """
    Stack feature vectors with a context window, mirroring at the boundaries.

    :param features: A 2D numpy array of shape (time_steps, feature_dim)
    :param n: Current time index
    :param context: Number of frames on each side of the current frame to include
    :return: A 1D numpy array containing the stacked feature vector
    """
    # Initialize a list to hold the context frames
    stacked_features = []
    num_frames = len(features)
    
    # Gather context frames, mirroring at the boundaries
    for i in range(n - context, n + context + 1):
        if i < 0:
            stacked_features.append(features[abs(i)])
        elif i >= num_frames:
            stacked_features.append(features[num_frames - 2 - (i - num_frames)])
        else:
            stacked_features.append(features[i])
    
    # return np.array(stacked_features)
    return np.concatenate(stacked_features)

def per_speaker_normalizer(features):
    # from sklearn.preprocessing import StandardScaler

    mean_feat = np.mean(features, axis=0)
    std_feat = np.std(features, axis=0)
    normalized_feat = (features - mean_feat) / std_feat

    return normalized_feat

def calc_dyn_feat(dataset):
    for data in tqdm(dataset):
        lmfcc_features = data['lmfcc']
        acoustic_context_features = []
        for n_idx in range(len(lmfcc_features)):
            new_features = stack_features(lmfcc_features, n_idx)
            acoustic_context_features.append(new_features)
        data['dyn_feat'] = np.array(acoustic_context_features)
    return dataset

def calc_normalization(dataset):
    for data in tqdm(dataset):
        lmfcc_features = data['lmfcc']
        normalized_lmfcc_feat = per_speaker_normalizer(lmfcc_features)
        data['normalized_lmfcc_feat'] = normalized_lmfcc_feat
    return dataset

def prepare_data(dataset, use_dynamic_feature, data_type='lmfcc'):
    num_frames = np.sum([len(data['lmfcc']) for data in dataset])
    if use_dynamic_feature:
        feat_dim = 13 * 7
    else:
        feat_dim = 13
    data_x = np.zeros((num_frames, feat_dim))
    data_y = np.zeros((num_frames))
    start_idx = 0
    for idx in range(len(dataset)):
        data = dataset[idx]

        if data_type == 'lmfcc':
            data_x[start_idx:start_idx+len(data['lmfcc'])] = data['normalized_lmfcc_feat']
        data_y[start_idx:start_idx+len(data['targets'])] = data['targets']

        start_idx = start_idx + len(data['normalized_lmfcc_feat'])

    data_x = data_x.astype('float32')

    # stateList = np.load('state_list.npy').tolist()
    # output_dim = len(stateList)
    # data_y = F.one_hot(torch.tensor(data_y, dtype=torch.int64), num_classes=output_dim)
    return data_x, data_y
    

def main():
   
    train_data = np.load('traindata.npz', allow_pickle=True)['traindata']
    test_data = np.load('testdata.npz', allow_pickle=True)['testdata']
 
    aa = 10

    #### 4.1 Target class definition 

#    phoneHMMs = np.load('lab2_models_all.npz',allow_pickle=True)['phoneHMMs'].item()
#    phones = sorted(phoneHMMs.keys())
#    nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
#    stateList = [  ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]

#    # save to file
#    state_array = np.array(stateList)
#    np.save('state_list.npy', state_array)
#    print("State list saved as an npy file.")

#    stateList = np.load('state_list.npy').tolist()

#    test = path2info('tidigits/disc_4.1.1/tidigits/train/woman/ag/1o2a.wav')


   #### 4.2 Forced alignment 
#    filename = 'tidigits/disc_4.1.1/tidigits/train/man/nw/z43a.wav'
#    samples, samplingrate = loadAudio(filename)
#    lmfcc = mfcc(samples)
#    wordTrans = list(path2info(filename)[2])
#    phoneTrans = words2phones(wordTrans, prondict)

#    viterbiStateTrans = forcedAlignment(lmfcc, phoneHMMs, phoneTrans)

#    target = [stateList.index(state) for state in viterbiStateTrans]

   # utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)
   # nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phoneTrans}
   # stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans
   #                for stateid in range(nstates[phone])]

   # obsloglik = log_multivariate_normal_density_diag(lmfcc, utteranceHMM['means'], utteranceHMM['covars'])

   # _, vpath = viterbi(obsloglik, np.log(utteranceHMM['startprob']), np.log(utteranceHMM['transmat']))

   # viterbiStateTrans = [stateTrans[state] for state in vpath]
   # frames2trans(viterbiStateTrans, outfilename='z43a.lab')


   #### 4.3 Feature extraction
   
   # # train data
   # traindata = []

   # for root, dirs, files in os.walk('tidigits/disc_4.1.1/tidigits/train'):
   #    for file in files:
   #       if file.endswith('.wav'):
   #          filename = os.path.join(root, file)
   #          samples, samplingrate = loadAudio(filename)
            
   #          lmfcc = mfcc(samples)
   #          mspec_res = mspec(samples)
   #          wordTrans = list(path2info(filename)[2])
   #          phoneTrans = words2phones(wordTrans, prondict)
            
   #          viterbiStateTrans = forcedAlignment(lmfcc, phoneHMMs, phoneTrans)

   #          targets = [stateList.index(state) for state in viterbiStateTrans]
            
   #          traindata.append({'filename': filename, 'lmfcc': lmfcc, 'mspec': mspec_res, 'targets': targets})
   
   # np.savez('traindata.npz', traindata=traindata)
   # print("finished")

   # # test data
   # testdata = []

   # for root, dirs, files in os.walk('tidigits/disc_4.2.1/tidigits/test'):
   #    for file in files:
   #       if file.endswith('.wav'):
   #          filename = os.path.join(root, file)
   #          samples, samplingrate = loadAudio(filename)
            
   #          lmfcc = mfcc(samples)
   #          mspec_res = mspec(samples)
   #          wordTrans = list(path2info(filename)[2])
   #          phoneTrans = words2phones(wordTrans, prondict)
            
   #          viterbiStateTrans = forcedAlignment(lmfcc, phoneHMMs, phoneTrans)

   #          targets = [stateList.index(state) for state in viterbiStateTrans]
            
   #          testdata.append({'filename': filename, 'lmfcc': lmfcc, 'mspec': mspec_res, 'targets': targets})
   
   # np.savez('testdata.npz', testdata=testdata)
   # print("finished")

   #### 4.4 Training and Validation Sets

    df = pd.DataFrame(list(train_data))

    df['speaker'] = df['filename'].apply(lambda x: x.split('/')[-2])
    df['gender'] = df['filename'].apply(lambda x: 'male' if x.split('/')[-3] == 'man' else 'female')

    # # print(df['gender'].value_counts())

    # # print("Unique speakers:", df['speaker'].nunique())
    # # print(df.groupby('gender')['speaker'].nunique())

    # aa = 10

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)

    for train_idx, val_idx in splitter.split(df, groups=df['speaker']):
        train_set = df.iloc[train_idx]
        val_set = df.iloc[val_idx]

    # gender_counts_train = train_set['gender'].value_counts(normalize=True)
    # gender_counts_val = val_set['gender'].value_counts(normalize=True)

    # print("Gender distribution in training set:\n", gender_counts_train)
    # print("Gender distribution in validation set:\n", gender_counts_val)

    # train_speakers = set(train_set['speaker'])
    # val_speakers = set(val_set['speaker'])

    # common_speakers = train_speakers.intersection(val_speakers)

    # if len(common_speakers) > 0:
    #    print("Warning: There are common speakers in both sets:", common_speakers)
    # else:
    #    print("Success: No common speakers in training and validation sets.")
    aa = 10

    #### 4.5 Acoustic Context (Dynamic Features)
    print(f'Calculating "4.5 Acoustic Context"')
    train_data = calc_dyn_feat(train_data)
    # test_data = calc_dyn_feat(test_data)

    #### 4.6 normalization
    print(f'Calculating "4.6 normalization"')
    train_data = calc_normalization(train_data)
    # test_data = calc_normalization(test_data)


    # Preparing data for training 
    use_dynamic_feature = True
    # use_dynamic_feature = False

    train_data_x, train_data_y = prepare_data(train_data, use_dynamic_feature=use_dynamic_feature)
    # test_data_x, test_data_y = prepare_data(test_data)

    np.savez('prepared_data.npz', 
             train_data_x=train_data_x,
             train_data_y=train_data_y,
    )

    # example = np.load('lab3_example.npz', allow_pickle=True)['example'].item()
   
    aa= 10

if __name__ == '__main__':
    main()