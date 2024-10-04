# DT2119, Lab 4 End-to-end Speech Recognition

import torch
from torch import nn
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm
from lab4_proto import dataProcessing, train_audio_transform, test_audio_transform, \
    intToStr, strToInt, greedyDecoder, levenshteinDistance
from loguru import logger
from pyctcdecode import build_ctcdecoder
from datetime import datetime
import os
import pandas as pd

'''
HYPERPARAMETERS
'''
hparams = {
	"n_cnn_layers": 3,
	"n_rnn_layers": 5,
	"rnn_dim": 512,
	"n_class": 29,
	"n_feats": 80,
	"stride": 2,
	"dropout": 0.1,
	"learning_rate": 5e-4, 
	"batch_size": 32, 
	"epochs": 20
}


'''
MODEL DEFINITION
'''
class CNNLayerNorm(nn.Module):
	"""Layer normalization built for cnns input"""
	def __init__(self, n_feats):
		super(CNNLayerNorm, self).__init__()
		self.layer_norm = nn.LayerNorm(n_feats)

	def forward(self, x):
		# x (batch, channel, feature, time)
		x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
		x = self.layer_norm(x)
		return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 

class ResidualCNN(nn.Module):
	"""Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
		except with layer norm instead of batch norm
	"""
	def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
		super(ResidualCNN, self).__init__()

		self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
		self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)
		self.layer_norm1 = CNNLayerNorm(n_feats)
		self.layer_norm2 = CNNLayerNorm(n_feats)

	def forward(self, x):
		residual = x  # (batch, channel, feature, time)
		x = self.layer_norm1(x)
		x = F.gelu(x)
		x = self.dropout1(x)
		x = self.cnn1(x)
		x = self.layer_norm2(x)
		x = F.gelu(x)
		x = self.dropout2(x)
		x = self.cnn2(x)
		x += residual
		return x # (batch, channel, feature, time)
		
class BidirectionalGRU(nn.Module):

	def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
		super(BidirectionalGRU, self).__init__()

		self.BiGRU = nn.GRU(
			input_size=rnn_dim, hidden_size=hidden_size,
			num_layers=1, batch_first=batch_first, bidirectional=True)
		self.layer_norm = nn.LayerNorm(rnn_dim)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		#logger.info('bi-gru, in:',x.shape)
		x = self.layer_norm(x)
		x = F.gelu(x)
		x, _ = self.BiGRU(x)
		x = self.dropout(x)
		#logger.info('bi-gru, out:',x.shape)
		return x

class SpeechRecognitionModel(nn.Module):
	"""Speech Recognition Model Inspired by DeepSpeech 2"""

	def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
		super(SpeechRecognitionModel, self).__init__()
		n_feats = n_feats//stride
		self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features

		# n residual cnn layers with filter size of 32
		self.rescnn_layers = nn.Sequential(*[
			ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
			for _ in range(n_cnn_layers)
		])
		self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
		self.birnn_layers = nn.Sequential(*[
			BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
							 hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
			for i in range(n_rnn_layers)
		])
		self.classifier = nn.Sequential(
			nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(rnn_dim, n_class),
			nn.LogSoftmax(dim=2)
		)

	def forward(self, x):
		x = self.cnn(x)
		x = self.rescnn_layers(x)
		sizes = x.size()
		x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
		x = x.transpose(1, 2) # (batch, time, feature)
		x = self.fully_connected(x)
		x = self.birnn_layers(x)
		x = self.classifier(x)
		return x

'''
ACCURACY MEASURES
'''
def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
	if ignore_case == True:
		reference = reference.lower()
		hypothesis = hypothesis.lower()

	ref_words = reference.split(delimiter)
	hyp_words = hypothesis.split(delimiter)
	edit_distance = levenshteinDistance(ref_words, hyp_words)
	ref_len = len(ref_words)

	if ref_len > 0:
		wer = float(edit_distance) / ref_len
	else:
		raise ValueError("empty reference string")  
	if wer > 1:
		aa = 10
	return wer

def cer(reference, hypothesis, ignore_case=False, remove_space=False):
	if ignore_case == True:
		reference = reference.lower()
		hypothesis = hypothesis.lower()

	join_char = ' '
	if remove_space == True:
		join_char = ''

	reference = join_char.join(filter(None, reference.split(' ')))
	hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

	edit_distance = levenshteinDistance(reference, hypothesis)
	ref_len = len(reference)
	if ref_len > 0:
		cer = float(edit_distance) / ref_len
	else:
		raise ValueError("empty reference string")
	return cer

'''
TRAINING AND TESTING
'''

def train(model, device, train_loader, criterion, optimizer, epoch):
	model.train()
	data_len = len(train_loader.dataset)
	logger.info("starting training")
	for batch_idx, _data in enumerate(train_loader):
		spectrograms, labels, input_lengths, label_lengths = _data 
		spectrograms, labels = spectrograms.to(device), labels.to(device)

		optimizer.zero_grad()
		# model output is (batch, time, n_class)
		output = model(spectrograms)  
		# transpose to (time, batch, n_class) in loss function
		loss = criterion(output.transpose(0, 1), labels, input_lengths, label_lengths)
		loss.backward()
		optimizer.step()
		if batch_idx % 10 == 0 or batch_idx == data_len:
			logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(spectrograms), data_len,
				100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, criterion, epoch, use_language_model=False, alpha=0.5, beta=1.0):
	logger.info('\nevaluating…')
	model.eval()
	test_loss = 0
	test_cer, test_wer = [], []
	if use_language_model:
		# specify alphabet labels as they appear in logits
		labels = [
			"'", " ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
			"m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
		]
		decoder = build_ctcdecoder(
			labels=labels,
			kenlm_model_path='wiki-interpolate.3gram.arpa',
			alpha=alpha,
			beta=beta,
		)
	with torch.no_grad():
		for I, _data in enumerate(tqdm(test_loader)):
			spectrograms, labels, input_lengths, label_lengths = _data 
			spectrograms, labels = spectrograms.to(device), labels.to(device)

			# model output is (batch, time, n_class)
			output = model(spectrograms)  
			# transpose to (time, batch, n_class) in loss function
			loss = criterion(output.transpose(0, 1), labels, input_lengths, label_lengths)
			test_loss += loss.item() / len(test_loader)

			# get target text
			decoded_targets = []
			for i in range(len(labels)):
				decoded_targets.append(intToStr(labels[i][:label_lengths[i]].tolist()))

			if use_language_model:
				decoded_preds = []
				for output_idx in range(output.shape[0]):
					text = decoder.decode(output[output_idx].cpu().detach().numpy())
					text = text.replace(" ", "_")
					decoded_preds.append(text)
			else:
				# get predicted text
				decoded_preds = greedyDecoder(output)

			# calculate accuracy
			for j in range(len(decoded_preds)):
				test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
				test_wer.append(wer(decoded_targets[j], decoded_preds[j], delimiter='_'))

	avg_cer = sum(test_cer)/len(test_cer)
	avg_wer = sum(test_wer)/len(test_wer)
	logger.info('Test set: Average loss: {:.4f}, Average CER: {:.4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))
	return avg_cer, avg_wer

'''
MAIN PROGRAM
'''
if __name__ == '__main__':
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--mode', help='train, test or recognize')
	argparser.add_argument('--model', type=str, help='model to load', default='')
	argparser.add_argument('wavfiles', nargs='*',help='wavfiles to recognize')

	args = argparser.parse_args()

	use_cuda = torch.cuda.is_available()
	torch.manual_seed(7)
	device = torch.device("cuda:0" if use_cuda else "cpu")

	train_dataset = torchaudio.datasets.LIBRISPEECH(".", url='train-clean-100', download=True)
	val_dataset = torchaudio.datasets.LIBRISPEECH(".", url='dev-clean', download=True)
	test_dataset = torchaudio.datasets.LIBRISPEECH(".", url='test-clean', download=True)

	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
	train_loader = data.DataLoader(dataset=train_dataset,
					batch_size=hparams['batch_size'],
					shuffle=True,
					collate_fn=lambda x: dataProcessing(x, train_audio_transform),
					**kwargs)

	val_loader = data.DataLoader(dataset=val_dataset,
					batch_size=hparams['batch_size'],
					shuffle=True,
					collate_fn=lambda x: dataProcessing(x, test_audio_transform),
					**kwargs)

	test_loader = data.DataLoader(dataset=test_dataset,
					batch_size=hparams['batch_size'],
					shuffle=False,
					collate_fn=lambda x: dataProcessing(x, test_audio_transform),
					**kwargs)

	model = SpeechRecognitionModel(
		hparams['n_cnn_layers'], 
		hparams['n_rnn_layers'], 
		hparams['rnn_dim'],
		hparams['n_class'], 
		hparams['n_feats'], 
		hparams['stride'], 
		hparams['dropout']
		).to(device)

	# write log into the file
	cur_version = datetime.now().strftime('%y%m%d-%H%M%S')
	os.makedirs('logs', exist_ok=True)

	if args.mode in ['train', 'grid_search']:
		logger.add(f"logs/log-{cur_version}.log")

	logger.info(model)
	print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

	optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
	criterion = nn.CTCLoss(blank=28).to(device)
	
	logger.info(args.mode)

	if args.model != '':
		model.load_state_dict(torch.load(args.model))
		logger.info(f'Load pre-trained model from "{args.model}"')

	best_wer = None
	use_language_model = True
	alpha = 0.5
	beta = 1.0

	if args.mode == 'train':
		for epoch in range(hparams['epochs']):
			train(model, device, train_loader, criterion, optimizer, epoch)
			avg_cer, avg_wer = test(model, device, val_loader, criterion, epoch, use_language_model, alpha, beta)
			if best_wer is None or avg_wer < best_wer:
				model_path = f'checkpoints/epoch-{epoch}-wer-{avg_wer:.3f}.pt'
				logger.info(f'Save the best model in "{model_path}"')
				torch.save(model.state_dict(), model_path)
				best_wer = avg_wer

	elif args.mode == 'test':
		avg_cer, avg_wer = test(model, device, test_loader, criterion, -1, use_language_model, alpha, beta)
		if use_language_model:
			logger.info(f"Use language model (alpha={alpha}, beta={beta}), avg_cer={avg_cer:.3f}, avg_wer={avg_wer:.3f}")
		else:
			logger.info(f"Not use language model, avg_cer={avg_cer:.3f}, avg_wer={avg_wer:.3f}")

	elif args.mode == 'recognize':
		for wavfile in args.wavfiles:
			waveform, sample_rate = torchaudio.load(wavfile, normalize=True)
			resample_rate = 16000
			if sample_rate != resample_rate:
				resampler = T.Resample(orig_freq=sample_rate, new_freq=resample_rate)
				waveform = resampler(waveform)
			spectrogram = test_audio_transform(waveform)
			input = torch.unsqueeze(spectrogram,dim=0).to(device)
			output = model(input)
			text = greedyDecoder(output)
			logger.info(f'wavfile: {wavfile}')
			logger.info(f'text: {text}')

	elif args.mode == 'grid_search':
		logger.info(f'Start grid searching for the language model...')
		grid_search_table = {}
		alpha_list = []
		beta_list = []
		lowest_wer = None
		for alpha in range(0, 11, 2):
			alpha = alpha / 10
			alpha_list.append(f'alpha_{alpha:.1f}')
		for beta in range(0, 11, 2):
			beta = beta / 10
			beta_list.append(f'beta_{beta:.1f}')
		for alpha in range(0, 11, 2):
			for beta in range(0, 11, 2):
				alpha = alpha / 10
				beta = beta / 10

				avg_cer, avg_wer = test(model, device, val_loader, criterion, -1, use_language_model, alpha, beta)
				grid_search_table[(f'alpha_{alpha:.1f}', f'beta_{beta:.1f}')] = avg_wer
				if lowest_wer is None or lowest_wer < avg_wer:
					lowest_wer = avg_wer
		logger.info(f'Lowest wer: {lowest_wer}')
		logger.info(f'results: \n {grid_search_table}')
		# Create a DataFrame to represent the table
		data = {alpha: [grid_search_table[(alpha, beta)] for beta in beta_list] for alpha in alpha_list}
		df = pd.DataFrame(data, index=beta_list)
		print(df)
