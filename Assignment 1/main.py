import sys
import json
import numpy as np


if __name__ == '__main__':
    path_in = sys.argv[1]
    path_js = sys.argv[2]

with open(path_in, 'r') as file:
    file_in = file.read()
    file_in = file_in.split("\n")
    for i in range(len(file_in)):
        file_in[i] = file_in[i].split()

y_matrix = []
for i in range(len(file_in)):
    y_matrix.append(file_in[i][-1])

x_matrix = []
for i in range(len(file_in)):
    x_matrix.append([1])
    x_matrix[i] += file_in[i][:len(file_in[0])-1]

x_matrix = np.array(x_matrix).astype(float)
y_matrix = np.array(y_matrix).astype(float)
x_matrix_transpose = x_matrix.transpose()

weights = np.matmul(np.matmul(np.linalg.inv(np.matmul(x_matrix_transpose,x_matrix)),x_matrix_transpose), y_matrix)

output_data = ""
for i in range(len(weights)):
    if i != 0:
        output_data += "\n"
    output_data += "{:.4f}".format(weights[i])

with open(path_js, 'r') as file:
    file_js = file.read()
file_js = json.loads(file_js)

weights2 = [0] * len(x_matrix[0])
weights2 = np.array(weights2)
weights2 = weights2.transpose()

learning_rate = file_js['learning rate']
num_iter = file_js['num iter']

n = num_iter
i = 0
while(i < n):
    loss = np.matmul((np.matmul(x_matrix,weights2)-y_matrix), x_matrix)
    weights2 = weights2 - (loss * learning_rate)
    i+=1

output_data += "\n"
for i in range(len(weights2)):
    output_data += "\n"
    output_data += "{:.4f}".format(weights2[i])

f = open("data/"+path_in[5:-3]+".out", "w")
f.write(output_data)
f.close()

