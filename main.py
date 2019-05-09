import models
import eval
import torch
import numpy as np
import datasets
import random
import argparse
import torch.utils.data as data


def parse_args():
    parser = argparse.ArgumentParser(description='Omniglot')

    parser.add_argument("--background_set_root", dest='background_set_root',
                        help="Root to the Background Set", default="images_background", type=str)
    parser.add_argument("--num_classes", dest='num_classes',
                        help="Number of classes to sample", default=964, type=int)
    parser.add_argument("--num_samples_train", dest='num_samples_train',
                        help="Number of samples for each class for training", default=16, type=int)
    parser.add_argument("--num_samples_val", dest='num_samples_val',
                        help="Number of samples for each class for validation", default=4, type=int)
    parser.add_argument("--num_workers", dest='num_workers',
                        help="Number of workers for Dataloaders", default=2, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)

    args = parse_args()
    # model = models.TripletNet()
    # avg_acc = eval.evaluate_all(model, prefix="all_runs")
    # print("Average Accuracy: {:.4f}".format(avg_acc))

    train_dict, val_dict = datasets.fetch_train_val(args.background_set_root)
    train_dataset = datasets.BasicDataset(train_dict)
    val_dataset = datasets.BasicDataset(val_dict)

    train_batch_sampler = datasets.BalancedBatchSampler(train_dataset.indices_dict,
                                                        num_classes=args.num_classes,
                                                        num_samples=args.num_samples_train,
                                                        len_dataset=len(train_dataset))

    train_data_loader = data.DataLoader(train_dataset, batch_sampler=train_batch_sampler,
                                        num_workers=args.num_workers, pin_memory=True)

    batch = next(train_data_loader.__iter__())

    anchors, positives, negatives, anc_labels, neg_labels = \
        datasets.gen_triplets_from_batch(batch,
                                         num_classes=args.num_classes,
                                         num_samples=args.num_samples_train)
    print(anchors.size())
    datasets.display_image(anchors[0], train_dataset.indices_to_labels[anc_labels[0].item()])
    datasets.display_image(positives[0], train_dataset.indices_to_labels[anc_labels[0].item()])
    datasets.display_image(negatives[0], train_dataset.indices_to_labels[neg_labels[0].item()])
