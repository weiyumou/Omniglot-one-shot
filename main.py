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


def parse_args():
    parser = argparse.ArgumentParser(description='Omniglot')

    parser.add_argument("--background_set_root", dest='background_set_root',
                        help="Root to the Background Set", default="images_background", type=str)
    parser.add_argument("--num_classes", dest='num_classes',
                        help="Number of classes to sample", default=964, type=int)
    parser.add_argument("--num_samples", dest='num_samples',
                        help="Number of samples for each class for training", default=20, type=int)
    parser.add_argument("--train_perct", dest='train_perct',
                        help="Percentage of the data for training", default=0.9, type=float)
    parser.add_argument("--num_workers", dest='num_workers',
                        help="Number of workers for Dataloaders", default=2, type=int)
    parser.add_argument("--model_dir", dest='model_dir',
                        help="Directory to save checkpoints", default="checkpoints", type=str)
    parser.add_argument("--eval_dir", dest='eval_dir',
                        help="Directory where evaluation data is stored", default="all_runs", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)

    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.TripletNet()
    if torch.cuda.device_count() > 1:
        print("{:d} GPUs are available".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    # avg_acc = eval.evaluate_all(device, model, prefix=args.eval_dir, num_runs=1)
    # print("Average Error Rate: {:.4f}".format(avg_acc))

    train_dict, val_dict = datasets.load_train_val(args.background_set_root, args.train_perct)
    train_dataset = datasets.BasicDataset(train_dict)
    val_dataset = datasets.BasicDataset(val_dict)

    num_samples = {"train": int(args.num_samples * args.train_perct),
                   "val": args.num_samples - int(args.num_samples * args.train_perct)}

    train_batch_sampler = datasets.BalancedBatchSampler(train_dataset.indices_dict,
                                                        num_classes=args.num_classes,
                                                        num_samples=num_samples["train"],
                                                        len_dataset=len(train_dataset))

    val_batch_sampler = datasets.BalancedBatchSampler(val_dataset.indices_dict,
                                                      num_classes=args.num_classes,
                                                      num_samples=num_samples["val"],
                                                      len_dataset=len(val_dataset))

    dataloaders = {"train": data.DataLoader(train_dataset, batch_sampler=train_batch_sampler,
                                            num_workers=args.num_workers, pin_memory=True),
                   "val": data.DataLoader(val_dataset, batch_sampler=val_batch_sampler,
                                          num_workers=args.num_workers, pin_memory=True)}

    # # criterion = nn.TripletMarginLoss(margin=10)
    criterion = losses.SlideLoss()
    optimiser = optim.Adam(model.parameters())
    batch_sizes = {"train": 256, "val": 256}

    # batch = next(dataloaders["val"].__iter__())
    # train_batch, test_batch, labels = datasets.gen_valset_from_batch(batch, args.num_classes, num_samples["val"])
    # err = eval.evaluate(device, model, train_batch, test_batch)
    # print(err)
    # print(train_batch.size())
    #
    # datasets.display_image(train_batch[1], val_dataset.indices_to_labels[labels[1].item()])
    # datasets.display_image(test_batch[1], val_dataset.indices_to_labels[labels[1].item()])

    # # anchors, positives, negatives, anc_labels, neg_labels = \
    # #     datasets.gen_triplets_from_batch(batch,
    # #                                      num_classes=args.num_classes,
    # #                                      num_samples=num_samples["val"])
    # # print(anchors.size())
    # # datasets.display_image(anchors[0], train_dataset.indices_to_labels[anc_labels[0].item()])
    # # datasets.display_image(positives[0], train_dataset.indices_to_labels[anc_labels[0].item()])
    # # datasets.display_image(negatives[0], train_dataset.indices_to_labels[neg_labels[0].item()])
    model, model_id = train.train_model(device, dataloaders, criterion, optimiser,
                                        args.model_dir, 300, args.num_classes,
                                        num_samples, batch_sizes, model=model)
    avg_err = eval.evaluate_all(device, model, prefix=args.eval_dir)
    print("Average Error Rate: {:.4f}".format(avg_err))
