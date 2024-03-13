import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
import torch
import torch.optim as optim
import torch.nn as nn
from IPython import display
from scipy.stats import multivariate_normal


mean = 0 # [0, 0]
cov = 1 #[[1, 0], [0, 1]]
var = multivariate_normal(mean=mean, cov=cov)
data = 0
var.cdf(data)
xx = np.linspace(-3,3, 100)
yy = var.pdf(xx)
plt.plot(xx, yy, '.', label = '1D Normal')
plt.xlabel('x')
plt.ylabel('PDF')
plt.legend()
plt.show()
X_train = np.linspace(-3,3, 1000)
y_train = var.pdf(X_train)
device = "cuda" if torch.cuda.is_available() else "cpu"
device
X_train = X_train.reshape(len(X_train), -1)
x_train_tensor = torch.from_numpy(X_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)
model = nn.Sequential(
    nn.Linear(1, 3),
    nn.ReLU(),
    nn.Linear(3, 1),
    nn.ReLU()
)
model =model.to(device)
# Linear Model
learning_rate = 0.05 # alpha
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
n_epoch = 5000
Losses = []
# Training
for t in range(n_epoch):
    # Feed forward to get the logits
    y_pred = model(x_train_tensor)
    # Compute the loss (BCE: Binary Cross Entropy Loss)
    loss = criterion(y_pred.T, y_train_tensor)
    # Dont accumulate previous gradients
    optimizer.zero_grad()
    print(f"[EPOCH]: {t}/{n_epoch}, [LOSS]: {loss.item():.6f}")
    if t % 100: Losses.append(loss.item())
    display.clear_output(wait=True)
    # Backward pass to compute the gradient
    # of loss w.r.t our learnable params.
    loss.backward()
    # Update params
    optimizer.step()
plt.plot(Losses)
x_test = np.linspace(-3,3, 100)
y_test = var.pdf(x_test)
x_test = x_test.reshape(len(x_test), -1)
x_test_tensor = torch.from_numpy(x_test).float().to(device)
y_pred = model(x_test_tensor)
y_pred = y_pred.flatten().cpu().detach().numpy()
plt.plot(x_test, y_test, '.', label = '1D Normal')
plt.plot(x_test, y_pred, '.', label = '1D Normal')
plt.xlabel('x')
plt.ylabel('PDF')
plt.legend()
plt.show()
model2 = nn.Sequential(
    nn.Linear(1, 3),
    nn.ReLU(),
    nn.Linear(3, 3),
    nn.ReLU(),
    nn.Linear(3, 1),
    nn.ReLU()
)
model2 = model2.to(device)
# Linear Model
learning_rate = 0.05 # alpha
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model2.parameters(), lr=learning_rate)
n_epoch = 5000
Losses = []
# Training
for t in range(n_epoch):
    # Feed forward to get the logits
    y_pred = model2(x_train_tensor)
    # Compute the loss (BCE: Binary Cross Entropy Loss)
    loss = criterion(y_pred.T, y_train_tensor)
    # Dont accumulate previous gradients
    optimizer.zero_grad()
    print(f"[EPOCH]: {t}/{n_epoch}, [LOSS]: {loss.item():.6f}")
    if t % 100: Losses.append(loss.item())
    display.clear_output(wait=True)
    # Backward pass to compute the gradient
    # of loss w.r.t our learnable params.
    loss.backward()
    # Update params
    optimizer.step()
plt.plot(Losses)
x_test = np.linspace(-3,3, 100)
y_test = var.pdf(x_test)
x_test = x_test.reshape(len(x_test), -1)
x_test_tensor = torch.from_numpy(x_test).float().to(device)
y_pred = model(x_test_tensor)
y_pred = y_pred.flatten().cpu().detach().numpy()
plt.plot(x_test, y_test, '.', label = '1D Normal')
plt.plot(x_test, y_pred, '.', label = '1D Normal')
plt.xlabel('x')
plt.ylabel('PDF')
plt.legend()
plt.show()