import torch
import time
import copy
import os
import tqdm
import math
import datasets
import eval


def train_model(device, dataloaders, criterion, optimiser, model_dir,
                num_epochs, num_classes, num_samples,
                batch_sizes, model, model_id=None):
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
                    anc = anchors[i].to(device)
                    pos = positives[i].to(device)
                    neg = negatives[i].to(device)

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

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Error: {:4f}'.format(best_val_err))
    model.load_state_dict(best_model_wts)
    checkpoint = {"model_state_dict": model.state_dict(),
                  "optimiser_state_dict": optimiser.state_dict(),
                  "last_epoch": last_epoch + num_epochs,
                  "best_val_err": best_val_err}
    torch.save(checkpoint, os.path.join(save_path, model_id + ".pt"))
    return model, model_id
