import os

import torch
import torchvision.transforms as transforms
from PIL import Image


def evaluate(model, folder, prefix):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    model = model.to(device)

    with open(os.path.join(prefix, folder, "class_labels.txt")) as file_hdl:
        test_files, train_files = zip(*map(str.split, file_hdl.readlines()))

    train_items = []
    for file in train_files:
        item = img_transforms(Image.open(os.path.join(prefix, file)))
        train_items.append(item)

    test_items = []
    for file in test_files:
        item = img_transforms(Image.open(os.path.join(prefix, file)))
        test_items.append(item)

    train_batch = torch.stack(train_items).to(device)  # torch.Size([20, 1, 105, 105])
    test_batch = torch.stack(test_items).to(device)

    with torch.no_grad():
        train_embeds = model(train_batch)
        test_embeds = model(test_batch)

    dist_matrix = calc_dist_matrix(train_embeds, test_embeds)
    preds = torch.argmax(dist_matrix, dim=1)
    labels = torch.tensor(range(train_batch.size(0)), device=device)
    corrects = torch.sum(preds == labels).item()
    acc = corrects / train_batch.size(0)
    return acc


def evaluate_all(model, num_runs=20, prefix="."):
    accuracy = 0.0
    for run in range(1, num_runs + 1):
        folder = "run" + str(run).zfill(2)
        acc = evaluate(model, folder, prefix)
        print("Run #{:d} Acc: {:.4f}".format(run, acc))
        accuracy += acc
    return accuracy / num_runs


def calc_dist_matrix(train_embeds, test_embeds):
    dot_prods = test_embeds @ train_embeds.transpose(0, 1)
    test_norms = torch.norm(test_embeds, dim=1, keepdim=True) ** 2
    train_norms = torch.norm(train_embeds, dim=1) ** 2
    dist_matrix = test_norms - 2 * dot_prods + train_norms
    return torch.sqrt(dist_matrix)
