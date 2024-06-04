import os
import csv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(2024) #set seed

from model import ShakePT, block_size, learning_rate, epochs

#########################################
############## Helpers ##################

### Tokenization  Helpers ###
def encode(string, dict):
    return [dict[s] for s in string]

def decode(idx, dict):
    return ''.join([dict[i] for i in idx])


### Data Loader Helper ###
def get_batch(split):
    data = train if split == 'train' else val
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

### Loss Expectation Helper ###
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#### Register Results Helper ####
def register(epoch):
    with open('results/losses.csv', 'a', newline='') as f:
        csx_writer = csv.writer(f)
        csx_writer.writerow([epoch, val_losses[-1]])

    plt.figure(figsize=(10, 5))
    plt.plot(range(0, epochs, 100),val_losses, label='Cross Entropy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/loss_plot.png')
    plt.close()

    torch.save(model.state_dict(), 'results/ShakePT.pth')

#########################################
#########################################

#### Data Processing ####
with open('data/shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
dict_ch = { ch:i for i,ch in enumerate(chars) }
dict_idx = { i:ch for i,ch in enumerate(chars) }
data = torch.tensor(encode(text, dict_ch), dtype=torch.long) # encode the character of the data
#########################

##########################
#### Global Variables ####
train_size = .8
batch_size = 64
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size = len(chars)
##########################
##########################

#### Data Splitting #####
train = data[:int(train_size*len(data))]
val = data[int(train_size*len(data)):]
#########################

if __name__ == "__main__":
    #### Create the result files ####
    if not os.path.exists('results'):
        os.makedirs('results')
    with open('results/losses.csv', 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['Epoch', 'Val_Loss'])
        

    model = ShakePT(vocab_size).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    val_losses = []
    for epoch in range(epochs):
        if epoch % 100 == 0:
            losses = estimate_loss()
            print(f"epoch {epoch}: train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
            val_loss = losses['val'].item()
            val_losses.append(val_loss)
            register(epoch)

        x, y = get_batch('train')

        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()



