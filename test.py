from argparse import ArgumentParser
import numpy as np
import torch
import shutil
import time
from torchvision.transforms import Compose
from src.load_data import Dataset, Resize, ToTensor
from src.model import CNN
from pathlib import Path
import cv2


def main():
    parser = ArgumentParser("Load Data")
    parser.add_argument("data_path", type=Path, help="path to dataset folder")
    parser.add_argument("model_path", type=Path, help="path to model")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 1

    transform = Compose([
        Resize(256, 256),
        ToTensor()
    ])

    test_dataset = Dataset(args.data_path, transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    model = CNN((256, 256), 2)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    for i, batch in enumerate(test_dataloader, start=1):
        img_batch, label_batch = batch["img"].to(device).float(), batch["label"].to(device).long()
        x = model(img_batch)
        x = torch.softmax(x, axis=-1)
        print(f"label: {label_batch.detach().cpu().numpy()}  -  prediction: {x.detach().cpu().numpy()}")

        image = np.transpose(img_batch[0].detach().cpu().numpy(), (1, 2, 0))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        while True:
            cv2.imshow("Image", image)
            key = cv2.waitKey(10)
            if key == ord("q"):
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    main()
