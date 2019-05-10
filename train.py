import torch
import time
import copy
import os
import tqdm
import math
import datasets


def train_model(dataloaders, criterion, optimiser, model_dir,
                num_epochs, num_classes, num_samples,
                batch_sizes, model_id=None, model=None):
    last_epoch = 0
    best_val_loss = math.inf

    if model_id is None and model is not None:
        model_id = str(int(time.time()))
        save_path = os.path.join(model_dir, model_id)
        os.makedirs(save_path, exist_ok=True)
    elif model_id is not None and model is None:
        save_path = os.path.join(model_dir, model_id)
        checkpoint = torch.load(os.path.join(save_path, model_id + ".pt"))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        last_epoch = checkpoint['epoch']
        best_val_loss = checkpoint["best_val_loss"]
    else:
        raise ValueError("Invalid model_id or model")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + last_epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            total = 0
            for batch in dataloaders[phase]:
                anchors, positives, negatives, *_ = \
                    datasets.gen_triplets_from_batch(batch,
                                                     num_classes=num_classes,
                                                     num_samples=num_samples[phase])
                anchors = anchors.split(batch_sizes[phase], dim=0)
                positives = positives.split(batch_sizes[phase], dim=0)
                negatives = negatives.split(batch_sizes[phase], dim=0)

                for i in range(len(anchors)):
                    anc = anchors[i].to(device)
                    pos = positives[i].to(device)
                    neg = negatives[i].to(device)

                    with torch.set_grad_enabled(phase == "train"):
                        anc_outs = model(anc)
                        pos_outs = model(pos)
                        neg_outs = model(neg)
                        loss = criterion(anc_outs, pos_outs, neg_outs)

                    if phase == "train":
                        optimiser.zero_grad()
                        loss.backward()
                        optimiser.step()

                    running_loss += loss.item() * anc.size(0)
                    total += anc.size(0)
                    break
                break

            epoch_loss = running_loss / total
            print("{} Loss: {:.4f}".format(phase, epoch_loss))

            if phase == "val" and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_val_loss))
    model.load_state_dict(best_model_wts)
    checkpoint = {"model_state_dict": model.state_dict(),
                  "optimiser_state_dict": optimiser.state_dict(),
                  "epoch": last_epoch + num_epochs,
                  "best_val_loss": best_val_loss}
    torch.save(checkpoint, os.path.join(save_path, model_id + ".pt"))
    return model, model_id
