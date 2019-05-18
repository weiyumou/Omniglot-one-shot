import models
import eval
import torch
import numpy as np
import datasets
import random
import argparse
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import train
import losses
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Omniglot')

    parser.add_argument("--background_set_root", dest='background_set_root',
                        help="Root to the Background Set", default="images_background", type=str)
    parser.add_argument("--train_perct", dest='train_perct',
                        help="Percentage of the data for training", default=0.9, type=float)
    parser.add_argument("--num_workers", dest='num_workers',
                        help="Number of workers for Dataloaders", default=2, type=int)
    parser.add_argument("--model_dir", dest='model_dir',
                        help="Directory to save checkpoints", default="checkpoints", type=str)
    parser.add_argument("--eval_dir", dest='eval_dir',
                        help="Directory where evaluation data is stored", default="all_runs", type=str)
    parser.add_argument("--batch_size_train", dest='batch_size_train',
                        help="Batch size for training", default=256, type=int)
    parser.add_argument("--batch_size_val", dest='batch_size_val',
                        help="Batch size for validation", default=256, type=int)
    parser.add_argument("--num_epochs", dest='num_epochs',
                        help="Number of epochs to run", default=200, type=int)
    parser.add_argument("--model_id", dest='model_id',
                        help="Specific model to train", default=None, type=str)
    parser.add_argument("--log_dir", dest='log_dir',
                        help="Directory to store logs", default="logs", type=str)
    parser.add_argument("--deterministic", dest='deterministic',
                        help="Whether to set random seed", default=False, action="store_true")
    parser.add_argument("--num_ways", dest='num_ways',
                        help="Number of ways of classification", default=20, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.deterministic:
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        np.random.seed(0)
        random.seed(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model = models.TripletNet().to(device)
    model = models.TripletNetWithFC().to(device)
    # model = models.MetricNet().to(device)
    # model = models.ResnetModel().to(device)
    # model = models.BrendenNet().to(device)
    # model = models.ResiduleNet().to(device)

    if torch.cuda.device_count() > 1:
        print("{:d} GPUs are available".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    optimiser = optim.Adam(model.parameters(), lr=5e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.95)
    criterion = losses.SlideLoss()

    train_dict, val_dict = datasets.load_train_val(args.background_set_root, args.train_perct)
    train_dataset = datasets.BasicDataset(train_dict)
    val_dataset = datasets.BasicDataset(val_dict)

    batch_sizes = {"train": args.batch_size_train, "val": args.batch_size_val}

    train_triplet_sampler = datasets.TripletSampler(train_dataset, shuffle=True)
    train_batch_sampler = datasets.TripletBatchSampler(train_triplet_sampler,
                                                       batch_size=batch_sizes["train"] * 3)
    triplet_dataloaders = {"train": data.DataLoader(train_dataset, batch_sampler=train_batch_sampler,
                                                    num_workers=args.num_workers, pin_memory=True)}

    pair_dataloaders = {"val": data.DataLoader(val_dataset, batch_size=2 * args.num_ways,
                                               shuffle=False, num_workers=args.num_workers,
                                               pin_memory=True, drop_last=True)}

    model, model_id, checkpoint = train.train_model(device, triplet_dataloaders, pair_dataloaders,
                                                    criterion, optimiser, scheduler, args.model_dir,
                                                    args.log_dir, args.num_epochs, model,
                                                    train.triplet_model_forward, eval.triplet_evaluate,
                                                    model_id=args.model_id)
    avg_err = eval.evaluate_all(device, model, eval.triplet_evaluate, prefix=args.eval_dir,
                                model_id=args.model_id, model_dir=args.model_dir)

    print("Average Error Rate: {:.4f}".format(avg_err))

    for phase in checkpoint["epoch_losses"]:
        epoch_losses = checkpoint["epoch_losses"][phase]
        plt.figure()
        plt.plot(range(len(epoch_losses)), epoch_losses)
        plt.title("Epoch Losses for {}".format(phase))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

    for phase in checkpoint["epoch_errors"]:
        epoch_errors = checkpoint["epoch_errors"][phase]
        plt.figure()
        plt.plot(range(len(epoch_errors)), epoch_errors)
        plt.title("Epoch Errors for {}".format(phase))
        plt.xlabel("Epoch")
        plt.ylabel("Error")

    plt.show()
