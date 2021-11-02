
import json, argparse, sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import os
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np 
sys.path.append('src')
sns.set_theme()


def x_y_data(ub, lb):
    mid_val = (ub + lb)/2
    x, y = np.meshgrid(np.linspace(-1,1,10),np.linspace(-1,1,10))
    return mid_val, x,y

def minmax_scale(scale, max_value, min_value):
    return((scale-min_value)/(max_value-min_value))

def reverse_scale(scale, max_value, min_value):
    return (scale * (max_value - min_value)) + min_value

def generate_data(ub, lb):
    x_2 = []
    y_2 = []
    mid_val = (ub + lb)/2
    
    for i in np.arange(lb,ub,0.1):
        init_x = i
        init_y = i
        for j in range(2000):
            if init_x > ub or init_x < lb:
                break
            if init_y > ub or init_y < lb:
                break
            x_2.append(init_x)
            y_2.append(init_y)
            init_x = (u(init_x, init_y)*0.01)+init_x
            init_y = (v(init_x, init_y)*0.01)+init_y
        init_x = -i
        init_y = i
        for j in range(2000):
            if init_x > ub or init_x < lb:
                break
            if init_y > ub or init_y < lb:
                break
            x_2.append(init_x)
            y_2.append(init_y)
            init_x = (u(init_x, init_y)*0.01)+init_x
            init_y = (v(init_x, init_y)*0.01)+init_y
        init_x = i
        init_y = -i
        for j in range(2000):
            if init_x > ub or init_x < lb:
                break
            if init_y > ub or init_y < lb:
                break
            x_2.append(init_x)
            y_2.append(init_y)
            init_x = (u(init_x, init_y)*0.01)+init_x
            init_y = (v(init_x, init_y)*0.01)+init_y
        init_x = i
        init_y = mid_val
        for j in range(2000):
            if init_x > ub or init_x < lb:
                break 
            if init_y > ub or init_y < lb:
                break
            x_2.append(init_x)
            y_2.append(init_y)
            init_x = (u(init_x, init_y)*0.01)+init_x
            init_y = (v(init_x, init_y)*0.01)+init_y
        
        init_x = mid_val
        init_y = i
        for j in range(2000):
            if init_x > ub or init_x < lb:
                break
            if init_y > ub or init_y < lb:
                break
            x_2.append(init_x)
            y_2.append(init_y)
            init_x = (u(init_x, init_y)*0.01)+init_x
            init_y = (v(init_x, init_y)*0.01)+init_y
        
        init_x = mid_val
        init_y = -i
        for j in range(2000):
            if init_x > ub or init_x < lb:
                break
            if init_y > ub or init_y < lb:
                break
            x_2.append(init_x)
            y_2.append(init_y)
            init_x = (u(init_x, init_y)*0.01)+init_x
            init_y = (v(init_x, init_y)*0.01)+init_y
        
        init_x = -i
        init_y = mid_val
        for j in range(2000):
            if init_x > ub or init_x < lb:
                break
            if init_y > ub or init_y < lb:
                break
            x_2.append(init_x)
            y_2.append(init_y)
            init_x = (u(init_x, init_y)*0.01)+init_x
            init_y = (v(init_x, init_y)*0.01)+init_y
    return x_2, y_2

def create_dataset(x,y):
    x_2_label = []
    y_2_label = []
    for i in range(1, len(x)):
        x_2_label.append(x[i])
        y_2_label.append(y[i])
    x = x[:len(x_2_label)]
    y = y[:len(x_2_label)]
    
    scale_x_feature = minmax_scale(x,max(x), min(x))
    scale_y_feature = minmax_scale(y,max(y), min(y))
    
    features = np.array([scale_x_feature, scale_y_feature]).transpose()
    
    scale_x_label = minmax_scale(x_2_label,max(x_2_label), min(x_2_label))
    scale_y_label = minmax_scale(y_2_label,max(y_2_label), min(y_2_label))
    
    labels = np.array([scale_x_label,scale_y_label]).transpose()
    
    max_x_label = max(x_2_label)
    max_y_label = max(y_2_label)
    min_x_label = min(x_2_label)
    min_y_label = min(y_2_label)
    
    features = features.reshape(features.shape[0],1,2)
    labels = labels.reshape(features.shape[0],1,2)
    return features, labels, max_x_label, max_y_label, min_x_label, min_y_label

def predict(n_tests):
    rand_x = random.sample(np.arange(lb, ub, 0.1).tolist(), n_tests)
    rand_y = random.sample(np.arange(lb, ub, 0.1).tolist(), n_tests)
    plt.quiver(x,y,u(x,y),v(x,y))
    for j in range(n_tests): 
        init_x = rand_x[j]
        init_y = rand_y[j]
        
        init_x = minmax_scale(init_x, max_x_label, min_x_label)
        init_y = minmax_scale(init_y, max_y_label, min_y_label)
        x_3 = []
        y_3 = []
        
        for j in range(100):
            if init_x > 1 or init_x < 0:
                break
            if init_y > 1 or init_y < 0:
                break
            x_3.append(init_x)
            y_3.append(init_y)
            x_temp = 0
            y_temp = 0
            y_temp = model.predict(np.array([init_x,init_y]).reshape(1,1,2))[0][0][1]
            x_temp = model.predict(np.array([init_x,init_y]).reshape(1,1,2))[0][0][0]
            init_y = y_temp
            init_x = x_temp
            x_rever = reverse_scale(np.array(x_3), max_x_label, min_x_label)
            y_rever = reverse_scale(np.array(y_3),max_y_label, min_y_label)
            plt.plot(x_rever, y_rever,marker='o', markevery = [0])

    file = result_path + '/fig'
    i = 1
    while os.path.exists(file + str(i)+'.pdf'):
        i += 1
    plt.savefig(result_path +'/fig'+str(i)+'.pdf')
    plt.close()
    





if __name__ == '__main__':

    # Command line arguments
    parser = argparse.ArgumentParser(description='ODE Solver')
    parser.add_argument('--param', help='filename for json atributes', metavar='param.json')
    parser.add_argument('-v', help='verbosity (default: 1)', metavar='N')
    parser.add_argument('--res-path', help='path to save the test plots at', metavar='results')
    parser.add_argument('--x-field', help='expression of the x-component of the vector field', metavar='x**2')
    parser.add_argument('--y-field', help='expression of the y-component of the vector field', metavar='y**2')
    parser.add_argument('--lb', help='lower bound for initial conditions', metavar='LB')
    parser.add_argument('--ub', help='upper bound for initial conditions', metavar='UB')
    parser.add_argument('--n-tests', help='number of test trajectories to plot', metavar='N_TESTS')
    args = parser.parse_args()

    verbosity = args.v

    with open(args.param) as paramfile:
        param = json.load(paramfile)
        
    learning_rate = param['learning rate']
    epochs = param['num epochs']
    BatchSize = param['Batch Size']
    
    verbosity = args.v
    result_path = args.res_path
    x_field = args.x_field
    y_field = args.y_field
    lb = float(args.lb)
    ub = float(args.ub)
    ntests = int(args.n_tests)

    def u(x, y):
        u = eval(x_field)
        return u

    def v(x, y):
        v = eval(y_field)
        return v

    mid_val, x, y = x_y_data(ub, lb)
    x_2, y_2 = generate_data(ub, lb)
    features, labels, max_x_label, max_y_label, min_x_label, min_y_label = create_dataset(x_2,y_2)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, 2), return_sequences=True))
    model.add(LSTM(50, activation='relu', return_sequences=True))
    model.add(Dense(2))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    history = model.fit(features, labels, validation_split=0.2, epochs=epochs, batch_size=BatchSize, verbose=verbosity)
    predict(ntests)



    

    