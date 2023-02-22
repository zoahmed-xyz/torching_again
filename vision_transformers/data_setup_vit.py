from going_modular import data_setup
from pathlib import Path
from torchvision import transforms


def create_dataloaders_vit():
    # Data preparation
    data_path = Path("../going_modular/data")
    image_path = data_path / "pizza_steak_sushi"
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    # Create image size (from Table 3 in the ViT paper)
    IMG_SIZE = 224

    manual_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # 1. Reshape all images to 224x224 (though some models may require different sizes)
        transforms.ToTensor(),  # 2. Turn image values to between 0 & 1
    ])

    # Create training and testing DataLoaders as well as get a list of class names
    train_dataloader_tl, test_dataloader_tl, class_names_tl = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                            test_dir=test_dir,
                                                                                            transform=manual_transforms,
                                                                                            # resize,
                                                                                            # convert images to between 0 & 1
                                                                                            batch_size=32)  # set mini-batch size- 32

    return train_dataloader_tl, test_dataloader_tl, class_names_tl


if __name__ == "__main__":
    train_dataloader, test_dataloader, class_names = create_dataloaders_vit()
    print(train_dataloader, test_dataloader, class_names)
