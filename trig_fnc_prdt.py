import torch.nn as nn
import torch
import math
import matplotlib.pyplot as plt
import random
import numpy as np

x = [i for i in range(1000)]

# CREATING RANDOM DATASET FOR EACH EPOCH
# IT RETURNS 100 SIN(X) VALUES , NEXT 100 SIN(X) VALUES , INDEX OF THOSE 100 SIN(X) VALUES FOR PLOTTING


def random_data_generator():
    y_temp = []
    x_temp = []
    index_start = random.choice(range(100, 890))
    for i in range(100):
        x_temp.append(x[i + index_start])
    x__ = x_temp.copy()
    for i in range(100):
        y_temp.append(((math.sin(x_temp[-1] + 0.1 * i)) * 30) + 5 * np.random.random(1))

    for i in range(100):
        x_temp[i] = math.sin(x_temp[i])
    return (
        torch.FloatTensor(x_temp).reshape((1, 1, 100)),
        torch.FloatTensor(y_temp).reshape(1, 1, 100),
        x__,
    )


# HERE THE MODEL HAS ONE LSTM LAYER AND ONE LINEAR LAYER


class lstm_net(nn.Module):
    def __init__(self):
        super(lstm_net, self).__init__()
        self.lstm_layer1 = nn.LSTM(100, 100)
        self.layer_output = nn.Linear(100, 100)

        # INITIALIZING OUR INITIAL HIDDEN VALUES (LONG TERM MEMORY VALUES)

        self.hidden_initialize = (torch.zeros((1, 1, 100)), torch.zeros((1, 1, 100)))

    def forward(self, x):
        x, hidden = self.lstm_layer1(x, self.hidden_initialize)
        x = self.layer_output(x)
        return x


my_model = lstm_net()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(my_model.parameters(), lr=0.001)
loss_history = []

# TRAINNING LOOP

for i in range(2000):
    x_, y_, _ = random_data_generator()

    optimizer.zero_grad()
    output = my_model(x_)
    loss = loss_function(output, y_)
    loss_history.append(loss)
    loss.backward()
    optimizer.step()

# PRININTG FINAL LOSS

print(loss_history[-1])
plt.plot(loss_history)
plt.show()

# PLOTTING OUT PREDICITON AND ACTUAL VALUES

x_, y_, x_index = random_data_generator()
model_prediction = my_model(x_)
y = []
for i in model_prediction[0][0]:
    y.append(i)
print(loss_function(model_prediction, y_))
plt.plot(x_index, y)
plt.plot(x_index, y_[0][0])
plt.show()
