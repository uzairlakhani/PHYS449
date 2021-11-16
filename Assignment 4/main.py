import json, argparse
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as func 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


## This formats the data from '+', '-' to +1, and -1 in a matrix
def data_format(path):
    data = open(path, "r").read()
    data1 = ''
    for i in range(len(data)):
        if data[i] == '+':
            data1 += '1 '
        elif data[i] == '-':
            data1 += '-1 '
        else:
            data1 += data[i]  
    data1 = data1.split('\n')
    N = len(np.fromstring(data1[0], dtype=int, sep=' '))
    
    data2 = np.empty((0,N))
    for i in range(len(data1)):
        if data1[i] == '':
            break
        new_arr = [np.fromstring(data1[i], dtype=int, sep=' ')]
        data2 = np.append(data2, new_arr, axis=0)
    data2 = data2.astype(np.float32)
    return data2, data2.tolist()


## This function creates a boltzmann distribution matrix first by multiplying xixj and then applying
## softmax function. Here log softmax is applied becuase Pytoch KL div loss takes in log softmax rather than just
## softmax. Applying just softmax gives negative loss. 
def boltzmann_dist(data):
    dist = data
    for i in range(len(dist)):
        for j in range(len(dist[0])):
            if j == len(dist[0])-1:
                dist[i][j] = data[i][j] * data[i][0]
            else:
                dist[i][j] = data[i][j] * data[i][j+1]
    dist = torch.tensor(dist)
    dist = func.log_softmax(dist, dim=1)
    return dist

## This function returns the data which is used in a model to create a distribution. Model has a softmax function
## which turns the returned data into distribution
def distribution(data):
    data = np.array(data).astype(np.float32)
    data = torch.tensor(data,requires_grad=False)
    return data

## This function trains the data
def training(model, dist, boltz_dist, learning_rate, num_epochs, display_epochs, verbose):
## reduction='batchmean' is used because it is specified in Pytorch documentation to use this.
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    obj_vals= []
    for epoch in range(num_epochs):
        obj_val = loss(boltz_dist, model.forward(dist))
        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
        obj_vals.append(obj_val.item())

        if ((epoch+1) % display_epochs == 0) and (verbose == 1):
            print ('Epoch [{}/{}]\tKL divergence Loss: {:.4f}'.format(epoch+1, num_epochs, obj_val.item()))
        elif ((epoch+1) % display_epochs == 0) and (verbose == 2):
            print ('Epoch [{}/{}]'.format(epoch+1, num_epochs))
        else:
  ##Default verbosity          
            if ((epoch+1) % display_epochs == 0):
                print ('Epoch [{}/{}]\tKL divergence Loss: {:.4f}'.format(epoch+1, num_epochs, obj_val.item()))

    
    parameters1=[]
    for p in model.parameters():
        parameters1.append(p.tolist())
    parameters1 = parameters1[-1]
    return obj_vals, parameters1, num_epochs

## This function outputs the result and plot and in the results folder. Output is found using the J_{i, j} found
## in model parameters. Couplers is found using coupler strengths J_{i, j} using the idea of magnetism. 
## When J_{i, j} > 0, interaction is ferromagnetic. Which means that the dipoles are in same direction (++ or --).
## When J_{i, j} < 0, interaction is antiferromagnetic. Which means that the dipoles are in opposite direction (+- or -+).
## Couplers are initialized to 1, and then adjusted based on how the follow coupler strengths on each coupler. 
def results(num_epochs, obj_vals, parameters):
    plt.plot(range(num_epochs), obj_vals, label= "KL divergence")
    plt.xlabel("Number of Epochs")
    plt.ylabel("KL Loss")
    plt.title("KL divergence Loss vs Number of Epochs")
    plt.savefig(result_path+'/fig.pdf')
    plt.close()
    
    couplers = np.full(len(parameters), 1)
    for i in range(len(couplers)):
        if i == len(couplers) - 1:
            if (parameters[i] < 0) and (couplers[i]==couplers[0]):
                couplers[i] *= -1
        else:
            if (parameters[i] < 0) and (couplers[i]==couplers[i+1]):
                couplers[i] *= -1
    output = "{"
    for i in range(len(couplers)):
        if i == len(couplers)-1:
            output += '('+str(i)+ ', '+str(0)+'): '+str(couplers[i])
        else:
            output += '('+str(i)+ ', '+str(i+1)+'): '+str(couplers[i])+', '
    output += "}"
    file =  open(result_path+'/out.txt', 'w')
    file.write(output)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() 
        self.fc1= nn.Linear(4, 4)

    def forward(self, x):
## softmax function turns it into a distribution, to be used in KL divergence
        h = func.softmax(self.fc1(x), dim=1)
        return h
    
    def reset(self):
        self.fc1.reset_parameters()
        


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description='Fully Visible Boltzmann Machine')
    parser.add_argument('--param', help='filename for json atributes', metavar='param.json')
    parser.add_argument('--data-path', help='path to get the data from', metavar='data')
    parser.add_argument('--res-path', help='path to save the plot and output at', metavar='results')
    parser.add_argument('-v', help='verbosity (default: 1). When verbosity is 1, KL Divergence shown.', metavar='N')
    args = parser.parse_args()

    with open(args.param) as paramfile:
        param = json.load(paramfile)

    learning_rate = param['learning rate']
    num_epochs = param['num epochs']
    display_epochs = param['display epochs']

    path_in = args.data_path
    result_path = args.res_path
    data = open(path_in, "r").read()
    data, data_list = data_format(path_in)
    boltz_dist = boltzmann_dist(data)
    dist = distribution(data_list)
    verbosity = int(args.v)

    model = Net().to(torch.device("cpu"))


    obj_vals, parameters, num_epochs = training(model, dist, boltz_dist, learning_rate, num_epochs, display_epochs, verbosity)
    results(num_epochs, obj_vals, parameters)
