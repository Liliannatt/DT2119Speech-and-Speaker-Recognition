import torchaudio
from torchaudio.transforms import MelSpectrogram, FrequencyMasking, TimeMasking
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence
from loguru import logger

# DT2119, Lab 4 End-to-end Speech Recognition


# Variables to be defined --------------------------------------
''' 
train-time audio transform object, that transforms waveform -> spectrogram, with augmentation
''' 
train_audio_transform = nn.Sequential(
    MelSpectrogram(n_mels=80),
    FrequencyMasking(freq_mask_param=15),
    TimeMasking(time_mask_param=35)
)
'''
test-time audio transform object, that transforms waveform -> spectrogram, without augmentation 
'''
test_audio_transform = nn.Sequential(
    MelSpectrogram(n_mels=80)
)

# Functions to be implemented ----------------------------------

def intToStr(labels):
    '''
        convert list of integers to string
    Args: 
        labels: list of ints
    Returns:
        string with space-separated characters
    '''
    int_to_char = {i+2: c for i, c in enumerate('abcdefghijklmnopqrstuvwxyz')}
    int_to_char[0] = "'"
    int_to_char[1] = "_"

    # Convert list of integers to text
    return ''.join([int_to_char[label] for label in labels])
    

def strToInt(text):
    '''
        convert string to list of integers
    Args:
        text: string
    Returns:
        list of ints
    '''
    text = text.lower()
    text = text.replace(' ', '_')
    char_to_int = {c: i + 2 for i, c in enumerate('abcdefghijklmnopqrstuvwxyz')}
    char_to_int["'"] = 0
    char_to_int["_"] = 1
    
    return [char_to_int[char] for char in text]

def dataProcessing(data, transform):
    '''
    process a batch of speech data
    arguments:
        data: list of tuples, representing one batch. Each tuple is of the form
            (waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)
        transform: audio transform to apply to the waveform
    returns:
        a tuple of (spectrograms, labels, input_lengths, label_lengths) 
        -   spectrograms - tensor of shape B x C x T x M 
            where B=batch_size, C=channel, T=time_frames, M=mel_band.
            spectrograms are padded the longest length in the batch.
        -   labels - tensor of shape B x L where L is label_length. 
            labels are padded to the longest length in the batch. 
        -   input_lengths - list of half spectrogram lengths before padding
        -   label_lengths - list of label lengths before padding
    '''
    
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    for (waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id) in data:
        # Transform the waveform to spectrogram
        spec = transform(waveform)
        spec = spec.squeeze(0).transpose(0, 1)
        
        # Convert the text to integer labels
        label = strToInt(utterance)
        
        # Append to the lists
        spectrograms.append(spec)
        labels.append(torch.tensor(label, dtype=torch.long))
        input_lengths.append(spec.shape[0] // 2)
        label_lengths.append(letn(label))
    
    # Pad the sequences
    spectrograms = pad_sequence(spectrograms, batch_first=True)
    labels = pad_sequence(labels, batch_first=True)
    
    # Rearrange dimensions
    spectrograms = spectrograms.unsqueeze(1).transpose(2, 3)
    
    return spectrograms, labels, input_lengths, label_lengths

def greedyDecoder(output, blank_label=28):
    '''
    decode a batch of utterances 
    arguments:
        output: network output tensor, shape B x T x C where B=batch_size, T=time_steps, C=characters
        blank_label: id of the blank label token
    returns:
        list of decoded strings
    '''
    str_list = []

    batch_size = output.shape[0]
    for batch_idx in range(batch_size):
        # Extract a sequence containing the most probable character for each time step
        res = torch.argmax(output[batch_idx], dim=1)

        # Merge any adjacent identical characters (get rid of the blank tokens)
        merged_ch = []
        for i in range(len(res)):
            if (len(merged_ch) == 0 or merged_ch[-1] != res[i]) and res[i] != blank_label:
                merged_ch.append(res[i].item())
        
        # Convert int list to strings
        str_list.append(intToStr(merged_ch))
    return str_list

def levenshteinDistance(ref,hyp):
    '''
    calculate levenshtein distance (edit distance) between two sequences
    arguments:
        ref: reference sequence
        hyp: sequence to compare against the reference
    output:
        edit distance (int)
    '''
    mat = torch.empty((len(ref) + 1, len(hyp) + 1))
    mat[0, :] = torch.arange(0, len(hyp) + 1)
    mat[:, 0] = torch.arange(0, len(ref) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                mat[i, j] = mat[i-1, j-1]
            else:
                mat[i, j] = min(mat[i-1, j] + 1, mat[i, j-1] + 1, mat[i-1, j-1] + 1)
    return mat[len(ref), len(hyp)]

def main():
    
    ### 3.2.3 verify
    example = torch.load('lab4_example.pt')
    inputs = example['data']
    expected_spectrograms = example['spectrograms']
    expected_labels = example['labels'].long() 
    expected_input_lengths = example['input_lengths']
    expected_label_lengths = example['label_lengths']
    
    spectrograms, labels, input_lengths, label_lengths = dataProcessing(inputs, test_audio_transform)
    
    logger.info("Generated spectrograms shape:", spectrograms.shape)
    logger.info("Expected spectrograms shape:", expected_spectrograms.shape)

    logger.info("Spectrograms match:", torch.allclose(spectrograms, expected_spectrograms, atol=1e-7))
    logger.info("Labels match:", torch.equal(labels, expected_labels))
    logger.info("Input lengths match:", input_lengths == expected_input_lengths)
    logger.info("Label lengths match:", label_lengths == expected_label_lengths)
        



if __name__ == '__main__':
    main()