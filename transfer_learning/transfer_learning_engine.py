import matplotlib.pyplot as plt
from timeit import default_timer as timer

import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchinfo import summary
from data_setup import create_dataloaders_transfer_learning
from going_modular import engine

torch.manual_seed(42)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Get dataloaders and class names
train_dataloader, test_dataloader, class_names = create_dataloaders_transfer_learning()

# Setup the model with pretrained weights and send it to the target device (torchvision v0.13+)
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights
model = torchvision.models.efficientnet_b0(weights=weights).to(device)
# Freeze all base layers in the "features" section of the model (the feature extractor)
# by setting requires_grad=False
for param in model.features.parameters():
    param.requires_grad = False

# Get the length of class_names (one output unit for each class)
output_shape = len(class_names)

# Recreate the classifier layer and seed it to the target device
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1280,
                    out_features=output_shape, # same number of output units as our number of classes
                    bias=True)).to(device)


# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


if __name__ == "__main__":
    start_time = timer()
    # Setup training and save the results
    results = engine.train(model=model,
                           train_dataloader=train_dataloader,
                           test_dataloader=test_dataloader,
                           optimizer=optimizer,
                           loss_fn=loss_fn,
                           epochs=5,
                           device=device)
    end_time = timer()
    print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")
