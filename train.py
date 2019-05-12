import torch
import time
import copy
import os
import tqdm
import math
import datasets
import eval
# import torch.utils.tensorboard as tensorboard


def train_model(device, dataloaders, criterion, optimiser, model_dir,
                num_epochs, num_classes, num_samples,
                batch_sizes, log_dir, model, model_id=None):
    last_epoch = 0
    best_val_err = math.inf

    if model_id is None:
        model_id = str(int(time.time()))
        save_path = os.path.join(model_dir, model_id)
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = os.path.join(model_dir, model_id)
        checkpoint = torch.load(os.path.join(save_path, model_id + ".pt"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        last_epoch = checkpoint["last_epoch"]
        best_val_err = checkpoint["best_val_err"]

    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_opt_params = copy.deepcopy(optimiser.state_dict())
    # writer = tensorboard.SummaryWriter(log_dir)
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + last_epoch, last_epoch + num_epochs - 1))
        print('-' * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            loss_total = 0
            running_err = 0.0
            err_total = 0
            for batch in dataloaders[phase]:
                anchors, positives, negatives, *_ = \
                    datasets.gen_triplets_from_batch(batch,
                                                     num_classes=num_classes,
                                                     num_samples=num_samples[phase])
                anchors = anchors.split(batch_sizes[phase], dim=0)
                positives = positives.split(batch_sizes[phase], dim=0)
                negatives = negatives.split(batch_sizes[phase], dim=0)

                if phase == "val":
                    train_batch, test_batch, _ = \
                        datasets.gen_valset_from_batch(batch, num_classes, num_samples[phase])

                for i in tqdm.tqdm(range(len(anchors)), desc="Triplet Batches"):
                    # grid = torchvision.utils.make_grid(anchors[i][:5])
                    # writer.add_image("Images", grid, 0)
                    # writer.add_graph(model, anchors[i])

                    anc = anchors[i].to(device)
                    pos = positives[i].to(device)
                    neg = negatives[i].to(device)

                    # if i == 0:
                    #     img_grid = torchvision.utils.make_grid([anc[0], pos[0], neg[0]])
                    #     datasets.display_image(img_grid)

                    with torch.set_grad_enabled(phase == "train"):
                        anc_out = model(anc)
                        pos_out = model(pos)
                        neg_out = model(neg)
                        loss = criterion(anc_out, pos_out, neg_out)

                    if phase == "train":
                        optimiser.zero_grad()
                        loss.backward()
                        optimiser.step()
                    else:
                        err = eval.evaluate(device, model, train_batch, test_batch)
                        running_err += err
                        err_total += 1

                    running_loss += loss.item() * anc.size(0)
                    loss_total += anc.size(0)

            epoch_loss = running_loss / loss_total
            print("{} Loss: {:.4f}".format(phase, epoch_loss))

            if phase == "val":
                epoch_err = running_err / err_total
                print("{} Error: {:.4f}".format(phase, epoch_err))

                if epoch_err < best_val_err:
                    best_val_err = epoch_err
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_opt_params = copy.deepcopy(optimiser.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Error: {:4f}'.format(best_val_err))
    model.load_state_dict(best_model_wts)
    optimiser.load_state_dict(best_opt_params)
    checkpoint = {"model_state_dict": model.state_dict(),
                  "optimiser_state_dict": optimiser.state_dict(),
                  "last_epoch": last_epoch + num_epochs,
                  "best_val_err": best_val_err}
    torch.save(checkpoint, os.path.join(save_path, model_id + ".pt"))
    # writer.close()
    return model, model_id


def train_triplet_model(device, triplet_dataloaders, pair_dataloaders,
                        criterion, optimiser, model_dir,
                        num_epochs, model, model_id=None):
    last_epoch = 0
    best_val_err = math.inf

    if model_id is None:
        model_id = str(int(time.time()))
        save_path = os.path.join(model_dir, model_id)
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = os.path.join(model_dir, model_id)
        checkpoint = torch.load(os.path.join(save_path, model_id + ".pt"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        last_epoch = checkpoint["last_epoch"]
        best_val_err = checkpoint["best_val_err"]

    best_model_wts = copy.deepcopy(model.state_dict())
    best_opt_params = copy.deepcopy(optimiser.state_dict())
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + last_epoch, last_epoch + num_epochs - 1))
        print('-' * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            loss_total = 0
            for triplet_batch, triplet_labels in tqdm.tqdm(triplet_dataloaders[phase], desc="Triplet Batches"):
                triplet_batch = torch.reshape(triplet_batch, (-1, 3, *triplet_batch.size()[1:]))
                triplet_labels = torch.reshape(triplet_labels, (-1, 3))
                anchors, positives, negatives = \
                    (torch.squeeze(x, dim=1) for x in torch.chunk(triplet_batch, 3, dim=1))
                anc_labels, pos_labels, neg_labels = \
                    (torch.squeeze(x, dim=1) for x in torch.chunk(triplet_labels, 3, dim=1))

                anchors = anchors.to(device)
                positives = positives.to(device)
                negatives = negatives.to(device)

                with torch.set_grad_enabled(phase == "train"):
                    anc_out = model(anchors)
                    pos_out = model(positives)
                    neg_out = model(negatives)
                    loss = criterion(anc_out, pos_out, neg_out)

                if phase == "train":
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

                running_loss += loss.item() * anc_labels.size(0)
                loss_total += anc_labels.size(0)

            epoch_loss = running_loss / loss_total
            print("{} Loss: {:.4f}".format(phase, epoch_loss))

            if phase == "val":
                running_err = 0.0
                err_total = 0
                for pair_batch, pair_labels in pair_dataloaders[phase]:
                    pair_batch = torch.reshape(pair_batch, (-1, 2, *pair_batch.size()[1:]))
                    pair_labels = torch.reshape(pair_labels, (-1, 2))
                    train_batch, test_batch = (torch.squeeze(x, dim=1) for x in torch.chunk(pair_batch, 2, dim=1))
                    train_labels, test_labels = (torch.squeeze(x, dim=1) for x in torch.chunk(pair_labels, 2, dim=1))
                    err = eval.evaluate(device, model, train_batch, test_batch)
                    running_err += err
                    err_total += 1

                epoch_err = running_err / err_total
                print("{} Error: {:.4f}".format(phase, epoch_err))

                if epoch_err < best_val_err:
                    best_val_err = epoch_err
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_opt_params = copy.deepcopy(optimiser.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Error: {:4f}'.format(best_val_err))
    model.load_state_dict(best_model_wts)
    optimiser.load_state_dict(best_opt_params)
    checkpoint = {"model_state_dict": model.state_dict(),
                  "optimiser_state_dict": optimiser.state_dict(),
                  "last_epoch": last_epoch + num_epochs,
                  "best_val_err": best_val_err}
    torch.save(checkpoint, os.path.join(save_path, model_id + ".pt"))
    return model, model_id
