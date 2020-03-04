import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import matplotlib.pyplot as plt
import random

x = [i / 10 for i in range(100)]


def random_data_generator():
    y_temp = []
    x_temp = []
    index_start = random.choice(range(10, 89))
    for i in range(10):
        x_temp.append(x[i + index_start])

    y_temp.append(math.sin(x_temp[-1] + 0.1))

    return (
        torch.FloatTensor(x_temp).reshape((1, 1, 10)),
        torch.FloatTensor(y_temp).reshape(1, 1, 1),
    )


class lstm_net(nn.Module):
    def __init__(self):
        super(lstm_net, self).__init__()
        # self.layer1 = nn.Linear(10,10)
        self.lstm_layer1 = nn.LSTM(10, 3)
        self.layer_output = nn.Linear(3, 1)
        self.hidden_initialzie = (torch.zeros((1, 1, 3)), torch.zeros((1, 1, 3)))

    def forward(self, x):
        # x = self.layer1(x)
        x, _ = self.lstm_layer1(x, self.hidden_initialzie)
        x = self.layer_output(x)
        return x


my_model = lstm_net()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(my_model.parameters(), lr=0.001)
loss_history = []

for i in range(100):
    optimizer.zero_grad()
    x_, y_ = random_data_generator()
    output = my_model(x_)
    loss = loss_function(output, y_)
    loss_history.append(loss)
    loss.backward()
    optimizer.step()

plt.plot(loss_history)
plt.show()

# x_ , y_ = random_data_generator()
#
# print(x_.reshape(1,1,x_.shape[0])[0].shape)
#
# lstm = nn.LSTM(3, 3,2)  # Input dim is 3, output dim is 3
# inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5
#
# # initialize the hidden state.
# hidden = (torch.randn(2, 1, 3),
#           torch.randn(2, 1, 3))
# for i in inputs:
#     # Step through the sequence one element at a time.
#     # after each step, hidden contains the hidden state.
#     out, hidden = lstm(i.view(1, 1, -1), hidden)
#
# print(inputs[0].shape)
# print(inputs[0].view(1, 1, -1).shape)
# print(hidden[0])
