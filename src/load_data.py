from pathlib import Path
import cv2
from argparse import ArgumentParser
import torch
import numpy as np

def main():
    parser = ArgumentParser("Load Data")
    parser.add_argument("data_path", type=Path, help="path to dataset folder")
    parser.add_argument("--limit", "--l", type=int, help="limit of the dataset")
    args = parser.parse_args()

    train_dataset = Dataset(args.data_path, Resize(256,256), args.limit)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)

    image, label = next(iter(train_dataloader))
    print(label)

    image = cv2.cvtColor(image[0].numpy(), cv2.COLOR_RGB2BGR)
    while True:
        cv2.imshow("Image", image)
        key = cv2.waitKey(10)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break

def load_dataset(folder:Path, limit:int = None):
    print ("load image")
    image_paths = list(folder.rglob("*.jpg"))
    labels = []
    for i, image_path in enumerate(image_paths):
        labels.append(1 if "cat" in image_path.name else 0)
        if limit and i > limit:
            break
    return image_paths, labels

class Resize(object):
    """ Resize the image in a sample to a given size. """
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        img = cv2.resize(img, (self.width, self.height))
        return {'img': img, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = img.transpose((2, 0, 1))
        return {'img': torch.from_numpy(img)/255.,
                'label': torch.from_numpy(np.asarray(label))}

class Dataset(torch.utils.data.Dataset):
    """Classification dataset."""
    def __init__(self, data_path: Path, transform=None, limit: int = None):
        """
        Args:
            data_path:
                Path to the root folder of the dataset.
                This folder is expected to contain subfolders for each class, with the images inside.
                It should also contain a "class.names" with all the classes
            transform (callable, optional): Optional transform to be applied on a sample.
            limit (int, optional): If given then the number of elements for each class in the dataset
                                   will be capped to this number
        """
        self.transform = transform
        self.image_paths, self.labels = load_dataset(data_path, limit=limit)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()
        img = cv2.imread(str(self.image_paths[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = int(self.labels[i])
        sample = {'img': img, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

if __name__ == "__main__":
    main()