import os
import requests
import zipfile
from pathlib import Path

# Setup path to data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"


# If the image folder doesn't exist, download it and prepare it...
def check_exists_image_dir(dir_path: Path = image_path):
    if dir_path.is_dir():
        print(f"{dir_path} directory exists.")
    else:
        print(f"Did not find {dir_path} directory, creating one...")
        dir_path.mkdir(parents=True, exist_ok=True)


def check_exists_data(dir_path: Path = image_path) -> bool:
    if not any(dir_path.iterdir()):
        print('The data directory is empty.')
        return False
    else:
        print('The directory is not empty.')
        return True


def download_data(dir_path: Path = data_path, image_data_path: Path = image_path):
    # Download pizza, steak, sushi data
    with open(dir_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Downloading pizza, steak, sushi data...")
        f.write(request.content)

    # Unzip pizza, steak, sushi data
    with zipfile.ZipFile(dir_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("Unzipping pizza, steak, sushi data...")
        zip_ref.extractall(image_data_path)

    # Remove zip file
    os.remove(data_path / "pizza_steak_sushi.zip")


if __name__ == "__main__":
    check_exists_image_dir()  # create image directory if it doesn't exist
    if check_exists_data():  # if data exists, do nothing
        print("Data is downloaded and exists")
    else:
        download_data()
