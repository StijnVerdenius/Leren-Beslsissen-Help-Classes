import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataloaders.toy import get_toy_loaders
from models.fc import FC
from training.nn_classification_trainer import NNClassificationTrainer

# get data
train_data, test_data = get_toy_loaders(length=1000, batch_size=32)

# analyse data before training
first_batch_data, firs_batch_labels = next(iter(test_data))
print("batch_shape:", first_batch_data.shape)
index_class_zero = firs_batch_labels == 0
index_class_one = firs_batch_labels == 1
data_zero = first_batch_data[index_class_zero].numpy()
data_one = first_batch_data[index_class_one].numpy()
plt.scatter(data_zero[:, 0], data_zero[:, 1], label="class_0")
plt.scatter(data_one[:, 0], data_one[:, 1], label="class_1")
plt.legend()
plt.title("first batch with ground-truth labels (before training)")
plt.show()

# determine device
device = "cuda" if torch.cuda.is_available() else "cpu"

# init train classes
loss = nn.CrossEntropyLoss()
model = FC(device=device, hidden_dim=16, n_classes=2, in_features=first_batch_data.shape[1])
optimizer = optim.SGD(model.parameters(), lr=0.005)  # alternatively: try optim.Adam

# do training
trainer = NNClassificationTrainer(model=model,
                                  loss=loss,
                                  optimizer=optimizer,
                                  device=device,
                                  train_loader=train_data,
                                  test_loader=test_data,
                                  epochs=100)
metrics, trained_model = trainer.train()

# plot results
for key in metrics[0]:
    values = []
    for i in range(len(metrics)):
        values.append(metrics[i][key])
    plt.plot(values)
    plt.title(key)
    plt.show()

# save results example
path = "./model.pickle"
torch.save(model.state_dict(), path)
with open("results.json", "w") as f:
    json.dump(metrics, f)

# loading trained model example
new_model_with_same_config = FC(device=device, hidden_dim=16, n_classes=2, in_features=first_batch_data.shape[1])
new_model_with_same_config.load_state_dict(torch.load(path))

# get number of parameters:
n_params = 0
for p in model.parameters():
    n_params += int(np.prod(p.shape))
print("parameter-count:", n_params)

# show they are identical
print("saved_model", model)
print("loaded_model", new_model_with_same_config)
same = True
for p1, p2 in zip(model.parameters(), new_model_with_same_config.parameters()):
    if p1.data.ne(p2.data).sum() > 0:
        same = False
        break
print("are they equal?", same)

exit()
