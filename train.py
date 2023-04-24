
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import lightning as L
import pandas as pd
from vad_dataset import LocalFileDataset, ChunkedDataset, SpeakDataset
import ast
from net1d import Net1D

SAMPLE_LENGTH = 512 # todo: this should be a parameter of the model...

def costume_collate_fn(batch):
    
    sample_list = []
    x_list      = []
    y_list      = []
    
    #Get Data
    for x, y, n in batch:

        x_list.append(x)
        y_list.append(y)
        
        #Für Zuordnung
        sample_list.append(torch.ones_like(y) * n)

    return torch.hstack(sample_list), torch.vstack(x_list), torch.hstack(y_list)

class Net1dTest(L.LightningModule):
	def __init__(self, n_input=1, n_output=1, stride=16, n_channel=32):
		super().__init__()
		
		self.net = Net1D(
			in_channels=1,
			base_filters=64,
			ratio=.5,
			filter_list = [64, 128, 128, 256, 256],
			m_blocks_list = [2, 2, 2, 3, 3],
			kernel_size=16,
			stride=2,
			groups_width=16,
			verbose=False,
			n_classes=1)

	def forward(self, x):
		#return x.view(-1,1)#F.log_softmax(x, dim=2)
		x = self.net(x)
		return x

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		_, x, y, = train_batch
		x = x.view(-1,1,512)
		#x = x.view(x.size(0), -1)
		z = self.forward(x)    		
		loss = F.cross_entropy(z, y.view(-1,1))
		self.log('train_loss', loss)
		return loss

	def validation_step(self, val_batch, batch_idx):
		_, x, y = val_batch
		x = x.view(-1,1,512)
		#x = x.view(x.size(0), -1)
		z = self.forward(x)    		
		loss = F.cross_entropy(z, y.view(-1,1))
		self.log('val_loss', loss)
		return loss

class SimpleVAD(L.LightningModule):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(SAMPLE_LENGTH, 512)
		self.fc2 = nn.Linear(512,64)
		self.fc3 = nn.Linear(64,64)
		self.fc4 = nn.Linear(64,64)
		self.fc5 = nn.Linear(64,1)

	def forward(self, x):
		out = torch.relu(self.fc1(x))
		out = torch.relu(self.fc2(out))
		out = torch.relu(self.fc3(out))
		out = torch.relu(self.fc4(out))
		out = self.fc5(out)
		return out

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		_, x, y, = train_batch
		#x = x.view(x.size(0), -1)
		z = self.forward(x)    		
		loss = F.binary_cross_entropy_with_logits(z, y.view(-1,1))
		self.log('train_loss', loss)
		return loss

	def validation_step(self, val_batch, batch_idx):
		_, x, y = val_batch
		#x = x.view(x.size(0), -1)
		z = self.forward(x)    		
		loss = F.binary_cross_entropy(z, y.view(-1,1))
		self.log('val_loss', loss)
		return loss

# data
#dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
#mnist_train, mnist_val = random_split(dataset, [55000, 5000])
#
#train_loader = DataLoader(mnist_train, batch_size=32)
#val_loader = DataLoader(mnist_val, batch_size=32)

#Konstante
SAMPLE_RATE = 16000

#Lädt CSV
train_csv = pd.read_csv(r"./data/train.csv")
test_csv  = pd.read_csv(r"./data/test.csv")
train_csv["voice"] = train_csv["voice"].apply(ast.literal_eval)
test_csv["voice"]  = test_csv["voice"].apply(ast.literal_eval)

#Dataset
file_dataset_train = LocalFileDataset(r"./data/SAMPLES GENERATED TRAIN",      data=train_csv, target_samplerate=SAMPLE_RATE)
file_dataset_test  = LocalFileDataset(r"./data/SAMPLES GENERATED TEST", data=test_csv,  target_samplerate=SAMPLE_RATE)

#SpeakDataset
speak_train_dataset_unchunked = SpeakDataset(file_dataset_train)

#ChunkedDataset
speak_train_dataset = ChunkedDataset(speak_train_dataset_unchunked)
dataloader_train = DataLoader(speak_train_dataset, batch_size=16, shuffle=True, collate_fn=costume_collate_fn, pin_memory=False, num_workers=8) 

# model
model = Net1dTest()

# training
trainer = L.Trainer(precision=16, limit_train_batches=1.0)
trainer.fit(model, dataloader_train)
    
