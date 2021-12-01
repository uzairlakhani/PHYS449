
import json, argparse, sys
import os
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as func 
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append('src')
sns.set_theme()


## function to load data
def load_data(path):
    data = np.loadtxt(path)
    np.random.shuffle(data)
    enc_data = []
    for i in range(len(data)):
        enc_data.append(data[i][:196])
    enc_data = np.array(enc_data)
    enc_data = enc_data.reshape(len(enc_data), 1, 14, 14)
## Normaliztion
    enc_data = torch.tensor(enc_data).float()/255
## returns shuffled data
    return enc_data

## KL loss function
def klLoss(model):
    mu = model.mu1
    var = model.var
    kl = (var**2 + mu**2 - torch.log(var) - 1/2).sum()
    return kl

## function to train variational autoencoder
def training(learning_rate, num_epochs, display_epochs, model, data, batchsize, verbose):

    learning_rate = learning_rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) 
    BCEloss1 = torch.nn.BCELoss(reduction='sum')
    num_epochs = int(num_epochs)
    display_epochs = int(display_epochs) 
    

    obj_vals = []
## data loader for batch sizes
    batch_data = DataLoader(data, batch_size=batchsize, shuffle=True)

    for epoch in range(num_epochs):
        for batch in batch_data:
##loss is BCE + KL
            obj_val = BCEloss1(model.forward(batch), batch) + klLoss(model)
            optimizer.zero_grad()
            obj_val.backward()
            optimizer.step()
        obj_vals.append(obj_val.item())
        
        if ((epoch+1) % display_epochs == 0) and (verbose == 1):
            print ('Epoch [{}/{}]\t Training Loss: {:.4f}'.format(epoch+1, num_epochs, obj_val.item()))

        elif ((epoch+1) % display_epochs == 0) and (verbose == 2):
            print ('Epoch [{}/{}]'.format(epoch+1, num_epochs))
        else:
  ##Default verbosity          
            if ((epoch+1) % display_epochs == 0):
                print ('Epoch [{}/{}]\t Training Loss: {:.4f}'.format(epoch+1, num_epochs, obj_val.item()))
            
    return obj_vals

## Variational Autoencoder model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() 
        
        self.enc1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding = 1, stride = 1)
        self.enc2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=2, padding = 0, stride = 2)
        
        self.fc1= nn.Linear(10 * 7 * 7, 50) 
        
        self.mu= nn.Linear(50, 5)
        self.log_var= nn.Linear(50, 5)
        
        self.fc2= nn.Linear(5, 50)
        self.fc3= nn.Linear(50, 10 * 7 * 7) 
        
        self.dec1 = nn.ConvTranspose2d(in_channels=10, out_channels=5, kernel_size=3, padding = 1, stride = 2)
        self.dec2 = nn.ConvTranspose2d(in_channels=5, out_channels=1, kernel_size=2, padding = 0, stride = 1)
        
        self.mu1 = 0
        self.var = torch.tensor(0)

        
    def forward(self, x):
        h = torch.sigmoid(self.enc1(x))
        h = torch.sigmoid(self.enc2(h))
        h = torch.flatten(h, start_dim=1)
        h = self.fc1(h)
        log_var = self.log_var(h)
        self.mu1 = self.mu(h)
        self.var = torch.exp(log_var)
        norm = torch.randn_like(self.var)
        z = self.mu1 + (self.var * norm)
        h = z
        h = self.fc2(h)
        h = self.fc3(h)
        unflatten = torch.nn.Unflatten(dim=1, unflattened_size=(10, 7, 7))
        h = unflatten(h)
        h = torch.sigmoid(self.dec1(h))
        h = torch.sigmoid(self.dec2(h))
        return h

## function to plot loss and decoded images. 
def results(model, loss, data, result_dir, num_output):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    num_epochs= len(loss)
    plt.plot(range(num_epochs), loss)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Number of Epochs")
    plt.savefig(result_dir+'/loss.pdf')
    plt.close()
## It is random because data was shuffled when loaded
    random_data = data[:num_output]
## multiplying 255 because it was normalized before
    data_output = model.forward(random_data).detach().numpy().reshape(num_output, 14, 14)*255

    for i in range(num_output):
        plt.imshow(data_output[i])
        plt.savefig(result_dir+'/'+str(i+1)+'.pdf')


if __name__ == '__main__':

    # Command line arguments
    parser = argparse.ArgumentParser(description='Variational Auto-Encoder')
    parser.add_argument('--param', help='filename for json atributes', metavar='param.json')
    parser.add_argument('-o', help='directory where the results are saved', metavar='results')
    parser.add_argument('-n', help='number of digit sample images', metavar='N_Images')
    parser.add_argument('-v', help='verbosity (default: 1). When verbosity is 1, Loss is shown.', metavar='N')
    args = parser.parse_args()


    with open(args.param) as paramfile:
        param = json.load(paramfile)
        
    learning_rate = param['learning rate']
    epochs = param['num epochs']
    BatchSize = param['Batch Size']
    display_epoch = param['display epochs']
    
    if args.v != None:
        verbosity = int(args.v)
    else:
        verbosity = args.v
    result_dir = args.o
    n_images = int(args.n)

    model = Net().to(torch.device("cpu"))
    data = load_data('data/even_mnist.csv')
    obj_vals = training(learning_rate, epochs, display_epoch, model, data, BatchSize, verbosity)
    results(model, obj_vals, data, result_dir, n_images)





    

    