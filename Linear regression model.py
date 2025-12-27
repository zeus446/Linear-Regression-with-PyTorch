import torch 
import matplotlib.pyplot as plt
import numpy as np
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = tensor = torch.arange(start,end,step).unsqueeze(dim=1)
y = weight*X + bias
train_split = int(len(X)*0.8)
x_train,y_train = X[:train_split],y[:train_split]
x_test,y_test = X[train_split:],y[train_split:]

def plot_pred(train_data = x_train,
              train_labels = y_train,
              test_data = x_test,
              test_labels = y_test,
              predictions = None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data,train_labels,c='b',label='Train data')
    plt.scatter(test_data,test_labels,c='g',label = 'Test data')
    if predictions is not None:
        plt.scatter(test_data,predictions,c='r',label='predictions')
    plt.legend()
    plt.show()


from torch import nn
class LinearRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float))

    def forward(self, x: torch.Tensor):
        return self.weights * x + self.bias

torch.manual_seed(42)
model_0 = LinearRegressor()

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01)
print(model_0.state_dict())

epochs = 200
epoch_values = []
test_loss_values = []
loss_values = []

#training loop

for epoch in range(epochs):
    model_0.train()

    y_pred = model_0(x_train)
    loss = loss_fn(y_pred, y_train)
    print(f"Loss:{loss}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#Testing

    model_0.eval()
    with torch.inference_mode():
        forward_pass = model_0(x_test)

        test_loss = loss_fn(forward_pass,y_test)

    if epoch%10==0:

        epoch_values.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)

        print(f"epoch: {epoch}|Loss:{loss}|Test Loss : {test_loss}")
        print(model_0.state_dict())

with torch.inference_mode():
    y_preds_new = model_0(x_test)
plot_pred(predictions=y_preds_new)
print(model_0.state_dict())

#plot the loss curve

plt.plot(epoch_values,np.array(torch.tensor(loss_values).numpy()),label = 'Training loss')
plt.plot(epoch_values,test_loss_values,label = "Test loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss graph")
plt.legend()
plt.show()


