import collections
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

import torch
import itertools

img_transforms = transforms.Compose([
    transforms.ToTensor()
])


def pil_loader(path):
    return Image.open(path)


def display_image(image_tensor, label=""):
    image_np = np.transpose(image_tensor.detach().cpu().numpy(), (1, 2, 0))
    plt.imshow(np.squeeze(image_np), cmap="gray")
    plt.xticks(())
    plt.yticks(())
    plt.text(2, 5, label)
    plt.show()


def load_train_val(root, train_perct):
    train_dict = collections.defaultdict(list)
    val_dict = collections.defaultdict(list)
    for alphabet in sorted(os.scandir(root), key=lambda x: x.name):
        if alphabet.is_dir(follow_symlinks=False):
            for char_class in sorted(os.scandir(alphabet.path), key=lambda x: x.name):
                if char_class.is_dir(follow_symlinks=False):
                    label = "_".join([alphabet.name, char_class.name])
                    for img_file in sorted(os.scandir(char_class.path),
                                           key=lambda x: x.name):
                        if img_file.is_file(follow_symlinks=False) and img_file.name.endswith(".png"):
                            train_dict[label].append(img_file.path)

                    val_index = int(len(train_dict[label]) * train_perct)
                    val_dict[label].extend(train_dict[label][val_index:])
                    train_dict[label][val_index:] = []
    return train_dict, val_dict


def gen_triplets_from_batch(batch, num_classes, num_samples):
    images, labels = batch
    image_clusters = torch.chunk(images, num_classes)  # tuple
    unique_labels = torch.unique(torch.stack(torch.chunk(labels, num_classes)), dim=1).reshape(-1)

    triplets = []
    for index, images in enumerate(image_clusters):
        for anc_pos_pair in itertools.combinations(images, 2):
            neg_class = random.randrange(num_classes)
            while neg_class == index:
                neg_class = random.randrange(num_classes)
            neg_idx = random.randrange(num_samples)
            triplets.append(anc_pos_pair + (image_clusters[neg_class][neg_idx],
                                            unique_labels[index], unique_labels[neg_class]))

    triplet_batch = list(zip(*triplets))
    # torch.Size([43380, 1, 105, 105])
    anchors, positives, negtives, anc_labels, neg_labels = tuple(map(torch.stack, triplet_batch))
    return anchors, positives, negtives, anc_labels, neg_labels


def gen_valset_from_batch(batch, num_classes, num_samples):
    if num_samples % 2 != 0:
        raise ValueError("num_samples must be even")
    images, labels = batch
    image_clusters = torch.chunk(images, num_classes, dim=0)
    image_pairs = torch.stack(image_clusters, dim=1)
    image_batches = torch.chunk(image_pairs, num_samples, dim=0)
    train_batch = torch.squeeze(torch.cat(image_batches[:len(image_batches) // 2], dim=1), dim=0)
    test_batch = torch.squeeze(torch.cat(image_batches[len(image_batches) // 2:], dim=1), dim=0)
    labels = torch.reshape(labels.reshape(-1, num_samples)[:, :num_samples // 2], (-1, ))
    return train_batch, test_batch, labels


class BalancedBatchSampler(data.Sampler):

    def __init__(self, indices_dict, num_classes, num_samples, len_dataset) -> None:
        super().__init__(indices_dict)
        self.indices_dict = indices_dict
        for label in self.indices_dict:
            random.shuffle(self.indices_dict[label])

        self.batch_size = num_classes * num_samples
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.len_dataset = len_dataset
        self.used_indices_count = collections.defaultdict(int)

    def __iter__(self):
        count = 0
        while count + self.batch_size <= self.len_dataset:
            classes = random.sample(self.indices_dict.keys(), self.num_classes)
            sel_indices = []
            for class_ in classes:
                indices = self.indices_dict[class_][self.used_indices_count[class_]:
                                                    self.used_indices_count[class_] + self.num_samples]
                sel_indices.extend(indices)
                self.used_indices_count[class_] += self.num_samples
                if self.used_indices_count[class_] + self.num_samples > len(self.indices_dict[class_]):
                    random.shuffle(self.indices_dict[class_])
                    self.used_indices_count[class_] = 0

            yield sel_indices
            count += self.batch_size

    def __len__(self):
        return self.len_dataset // self.batch_size


class BasicDataset(data.Dataset):

    def __init__(self, data_dict, loader=pil_loader, transform=img_transforms) -> None:
        super().__init__()
        self.data_dict = data_dict
        self.labels_to_indices = dict(zip(sorted(data_dict.keys()), range(len(data_dict))))
        self.indices_to_labels = dict(zip(range(len(data_dict)), sorted(data_dict.keys())))
        self.loader = loader
        self.transform = transform

        self.indices_dict = dict()
        curr_index = 0
        for label in self.data_dict:
            self.indices_dict[label] = list(range(curr_index,
                                                  curr_index + len(self.data_dict[label])))
            curr_index += len(self.data_dict[label])

    def __getitem__(self, index: int):
        curr_index = 0
        for label in self.data_dict:
            if index < curr_index + len(self.data_dict[label]):
                img = self.loader(self.data_dict[label][index - curr_index])
                if self.transform is not None:
                    img = self.transform(img)
                return img, self.labels_to_indices[label]
            curr_index += len(self.data_dict[label])

    def __len__(self) -> int:
        return sum((len(self.data_dict[label]) for label in self.data_dict))
