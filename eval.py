import os
import datasets
import torch
import torch.nn.functional as F


def triplet_evaluate(device, model, train_batch, test_batch):
    train_batch = train_batch.to(device)
    test_batch = test_batch.to(device)
    assert not model.training
    with torch.no_grad():
        train_embeds, *_ = model(train_batch)
        test_embeds, *_ = model(test_batch)

    dist_matrix = torch.stack([F.pairwise_distance(train_embeds, test_embeds[i], p=2)
                               for i in range(test_embeds.size(0))])
    preds = torch.argmin(dist_matrix, dim=1)
    labels = torch.tensor(range(train_batch.size(0)), device=device)
    corrects = torch.sum(preds == labels).item()
    acc = corrects / train_batch.size(0)
    return 1 - acc


# def metric_evaluate(device, model_dict, train_batch, test_batch):
#     triplet_model = model_dict["triplet"]
#     metric_model = model_dict["metric"]
#
#     train_batch = train_batch.to(device)
#     test_batch = test_batch.to(device)
#     with torch.no_grad():
#         train_batch = triplet_model(train_batch)
#         test_batch = triplet_model(test_batch)
#         concat_batch = torch.cat([torch.cat((train_batch, test_batch[i].expand_as(train_batch)), dim=1)
#                                   for i in range(test_batch.size(0))], dim=0)
#         dist_matrix = metric_model(concat_batch)
#
#     dist_matrix = dist_matrix.reshape(-1, train_batch.size(0))
#     preds = torch.argmin(dist_matrix, dim=1)
#     labels = torch.tensor(range(train_batch.size(0)), device=device)
#     corrects = torch.sum(preds == labels).item()
#     acc = corrects / train_batch.size(0)
#     return 1 - acc


def load_eval_images(folder, prefix, loader=datasets.pil_loader,
                     transform=datasets.img_transforms):
    with open(os.path.join(prefix, folder, "class_labels.txt")) as file_hdl:
        test_files, train_files = zip(*map(str.split, file_hdl.readlines()))

    train_items = []
    for file in train_files:
        item = transform(loader(os.path.join(prefix, file)))
        train_items.append(item)

    test_items = []
    for file in test_files:
        item = transform(loader(os.path.join(prefix, file)))
        test_items.append(item)

    train_batch = torch.stack(train_items)
    test_batch = torch.stack(test_items)

    return train_batch, test_batch


def evaluate_all(device, model, eval_forward, model_id=None,
                 model_dir=None, num_runs=20, prefix="."):
    if model_id is not None and model_dir is not None:
        # model = torch.nn.DataParallel(model)
        save_path = os.path.join(model_dir, model_id)
        checkpoint = torch.load(os.path.join(save_path, model_id + ".pt"),
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    error = 0.0
    run_errors = []
    for run in range(num_runs):
        folder = "run" + str(run + 1).zfill(2)
        train_batch, test_batch = load_eval_images(folder, prefix)
        err = eval_forward(device, model, train_batch, test_batch)
        run_errors.append(err)
        print("Run #{:d} Error Rate: {:.4f}".format(run, err))
        error += err
    return error / num_runs, run_errors


def calc_dist_matrix(train_embeds, test_embeds):
    dot_prods = test_embeds @ train_embeds.transpose(0, 1)
    test_norms = torch.norm(test_embeds, dim=1, keepdim=True) ** 2
    train_norms = torch.norm(train_embeds, dim=1) ** 2
    dist_matrix = test_norms - 2 * dot_prods + train_norms
    return torch.sqrt(dist_matrix)
