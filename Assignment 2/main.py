import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
import torch.nn.functional as func 
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
sys.path.append('src')

# Convolutional Neural Network model to be used to train the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() 
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding = 1, stride = 1)
        self.cnn2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding = 1, stride = 1)
        self.fc1= nn.Linear(10 * 14 * 14, 100) 
        self.fc2= nn.Linear(100, 5) 

    def forward(self, x):
        h = func.relu(self.cnn1(x))
        h = func.relu(self.cnn2(h))
        h = torch.flatten(h, start_dim=1)
        h = func.relu(self.fc1(h))
        y = self.fc2(h)
        return y
    
    def reset(self):
        self.cnn1.reset_parameters()
        self.cnn2.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

# function to load the data        
def load_data(path):
    data = np.loadtxt(path)
#shuffles the data
    np.random.shuffle(data)
    labels = []
    features = []
    for i in range(len(data)):
        labels.append(data[i][-1])
        features.append(data[i][:196])
    x_train = features[:len(features)-3000]
    x_test = features[len(features)-3000:]
    y_train = labels[:len(features)-3000]
    y_test = labels[len(features)-3000:]
    x_train = np.array(x_train)
    x_train = x_train.reshape(len(x_train), 1, 14, 14)
    x_test = np.array(x_test)
    x_test = x_test.reshape(len(x_test), 1, 14, 14)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test

# function to check the accuracy of the model
def accuracy(predict, targets):
    actual = targets.tolist()
    predict = predict.tolist()
    prediction = np.argmax(predict, axis = 1)
    accuracy = (actual==prediction).astype(int)
    accuracy = sum(accuracy.astype(int))/len(accuracy)*100
    return accuracy

# function to train the model 
def run(learning_rate, num_epochs, display_epochs, model, data_path):
    x_train, y_train, x_test, y_test = load_data(data_path)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate) 
    loss = torch.nn.CrossEntropyLoss(reduction='mean')
    num_epochs = int(num_epochs)
    display_epochs = int(display_epochs) 

    obj_vals = []
    cross_vals = []
    accuracy_train = []
    accuracy_test = []

    for epoch in range(num_epochs):
        inputs = torch.from_numpy(x_train).float()
        targets = torch.from_numpy(y_train).float()
        targets = targets/2
        targets = targets.type(torch.LongTensor)
        
        obj_val = loss(model.forward(inputs), targets)
        accuracy1 = accuracy(model.forward(inputs), targets)
        accuracy_train.append(accuracy1)
        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
        obj_vals.append(obj_val.item())
        
        if (epoch+1) % display_epochs == 0:
            print ('Epoch [{}/{}]\t Training Loss: {:.4f}'.format(epoch+1, num_epochs, obj_val.item()))
            print ('Epoch [{}/{}]\t Training Accuracy: {:.4f}%'.format(epoch+1, num_epochs, accuracy1))
        
        with torch.no_grad(): 
            inp = torch.from_numpy(x_test).float()
            targ = torch.from_numpy(y_test).float()
            targ = targ/2
            targ = targ.type(torch.LongTensor)
            cross_val = loss(model.forward(inp), targ)
            cross_vals.append(cross_val)
            accuracy2 = accuracy(model.forward(inp), targ)
            accuracy_test.append(accuracy2)
        
        if (epoch+1) % display_epochs == 0:
            print ('Epoch [{}/{}]\t Test Loss: {:.4f}'.format(epoch+1, num_epochs, cross_val.item()))
            print ('Epoch [{}/{}]\t Test Accuracy: {:.4f}%\n'.format(epoch+1, num_epochs, accuracy2))
        
    print('Final training loss: {:.4f}'.format(obj_vals[-1]))
    print('Final test loss: {:.4f}'.format(cross_vals[-1]))
    print('Final training accuracy: {:.4f}'.format(accuracy_train[-1]))
    print('Final test accuracy: {:.4f}'.format(accuracy_test[-1]))

    return obj_vals, cross_vals, accuracy_train, accuracy_test

# function to plot the results in the results folder
def plot_results(obj_vals, cross_vals, accuracy_train, accuracy_test):
    num_epochs= len(obj_vals)
    plt.plot(range(num_epochs), obj_vals, label= "Training loss", color="blue")
    plt.plot(range(num_epochs), cross_vals, label= "Test loss", color= "green")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Number of Epochs")
    plt.legend()
    plt.savefig('results/loss.pdf')
    plt.close()
    plt.plot(range(num_epochs), accuracy_train, label= "Training accuracy", color="blue")
    plt.plot(range(num_epochs), accuracy_test, label= "Test accuracy", color= "green")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of Epochs")
    plt.legend()
    plt.savefig('results/accuracy.pdf')
    plt.close()

# function to generate the results on the testing data in results folder
def report(obj_vals, cross_vals, accuracy_train, accuracy_test):
    test_loss = cross_vals[-1]
    accuracy_test = accuracy_test[-1]
    data = """The model is trainied on Convolutional Neural Network using Pytorch.
    \nFinal Loss on testing data is {0:.4f}\nFinal Accuracy on testing data is {0:.4f}%""".format(test_loss.item(), accuracy_test)
    file = open("results/report.txt", "w")
    file.write(data)
    file.close()



if __name__ == '__main__':
    path_js = sys.argv[1]
    with open(path_js, 'r') as file:
        file_js = file.read()
        file_js = json.loads(file_js)

    model = Net().to(torch.device("cpu"))
    model.reset()

    learning_rate = file_js['learning rate']

    num_epochs = file_js['num epochs']

    dispay_epochs = file_js['display epochs']

# training model
    obj_vals, cross_vals, accuracy_train, accuracy_test = run(learning_rate, num_epochs, dispay_epochs, model, 'data/even_mnist.csv')

# generating the results on testing data in results folder
    report(obj_vals, cross_vals, accuracy_train, accuracy_test)

# plotting results in result folder
    plot_results(obj_vals, cross_vals, accuracy_train, accuracy_test)
