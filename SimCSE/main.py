import pandas as pd
import torch as nn
import seaborn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
import numpy as np

from torch.utils.data import DataLoader, Dataset

from BertCL import BertCL
from CLDataset import CLDataset
from ContrastiveLoss import ContrastiveLoss

seed = 42
random.seed(seed)
np.random.seed(seed)
nn.manual_seed(seed)
nn.cuda.manual_seed_all(seed)

df = pd.read_csv('./training/indataset_sen_pos.csv')

train, test = train_test_split(df, test_size=0.8)

model = BertCL()
optimizer = nn.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2)

train_dataset = CLDataset(train)
test_dataset = CLDataset(test)

BATCH_SIZE = 8

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataset = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

running_loss = []

loss = ContrastiveLoss()

training_epochs = 10

for epoch in tqdm(range(training_epochs)):
    avg_epoch_loss = 0.0
    for batch_idx, batch in enumerate(train_dataloader):
        sentences, positives = batch

        optimizer.zero_grad()

        sentences_output = model(list(sentences))
        positives_output = model(list(positives))

        output = loss(sentences_output, positives_output)

        output.mean().backward()
        optimizer.step()

        avg_epoch_loss += avg_epoch_loss

    avg_epoch_loss /= BATCH_SIZE

    running_loss.append(avg_epoch_loss)

seaborn.lineplot(x=range(training_epochs), y=running_loss)
