from going_modular import data_setup
from pathlib import Path
from torchvision import transforms


def create_dataloaders_transfer_learning():
    # Data preparation
    data_path = Path("../going_modular/data")
    image_path = data_path / "pizza_steak_sushi"
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    # When using a pretrained model, it's important that your custom data going into the model is prepared in the same
    # way as the original training data that went into the model.
    # Create a transforms pipeline manually (required for torchvision < 0.13)
    manual_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # 1. Reshape all images to 224x224 (though some models may require different sizes)
        transforms.ToTensor(),  # 2. Turn image values to between 0 & 1
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                             std=[0.229, 0.224, 0.225])  # 4. A standard deviation of [0.229, 0.224, 0.225] (across each
        # colour channel),
    ])

    # Create training and testing DataLoaders as well as get a list of class names
    train_dataloader_tl, test_dataloader_tl, class_names_tl = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                   test_dir=test_dir,
                                                                                   transform=manual_transforms, # resize,
                                                                                   # convert images to between 0 & 1 and
                                                                                   # normalize them
                                                                                   batch_size=32) # set mini-batch size- 32

    return train_dataloader_tl, test_dataloader_tl, class_names_tl

if __name__ == "__main__":
    train_dataloader, test_dataloader, class_names = create_dataloaders_transfer_learning()
    print(train_dataloader, test_dataloader, class_names)
