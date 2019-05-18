import torch
import time
import copy
import os
import tqdm
import math
import collections
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import eval


def triplet_model_forward(model, criterion, anchors, positives, negatives):
    anc_out, conv3_out, conv2_out, conv1_out = model(anchors)
    pos_out, *_ = model(positives)
    neg_out, *_ = model(negatives)

    anc_pos = F.pairwise_distance(anc_out, pos_out, p=2, keepdim=True)
    anc_neg = F.pairwise_distance(anc_out, neg_out, p=2, keepdim=True)
    loss = criterion(anc_pos, anc_neg)
    return loss, anc_out, pos_out, neg_out, conv3_out, conv2_out, conv1_out


# def metric_model_forward(model_dict, criterion,
#                          anchors, positives, negatives, writer, epoch):
#     triplet_model = model_dict["triplet"]
#     anc_out = triplet_model(anchors)
#     pos_out = triplet_model(positives)
#     neg_out = triplet_model(negatives)
#
#     anc_pos = torch.cat((anc_out, pos_out), dim=1)
#     anc_neg = torch.cat((anc_out, neg_out), dim=1)
#     metric_model = model_dict["metric"]
#     anc_pos_out = metric_model(anc_pos)
#     anc_neg_out = metric_model(anc_neg)
#     loss = criterion(anc_pos_out, anc_neg_out)
#     return loss
# anc_pos = torch.cat((anchors, positives), dim=1)
# anc_neg = torch.cat((anchors, negatives), dim=1)
# anc_pos_out = model(anc_pos)
# anc_neg_out = model(anc_neg)
# loss = criterion(anc_pos_out, anc_neg_out)
# return loss


def adv_model_forward(model, criterion, anchors, positives, negatives):
    negatives.requires_grad_()
    loss, *_ = triplet_model_forward(model, criterion, anchors, positives, negatives)
    if loss.requires_grad:
        model.zero_grad()
        loss.backward()
    else:
        return loss
    with torch.no_grad():
        negatives = torch.clamp(negatives + 1e-2 * torch.sign(negatives.grad), min=0, max=1)
    return triplet_model_forward(model, criterion, anchors, positives, negatives)


def train_model(device, triplet_dataloaders, pair_dataloaders,
                criterion, optimiser, scheduler, model_dir, log_dir,
                num_epochs, model, model_forward, eval_forward, model_id=None):
    last_epoch = 0
    best_val_err = math.inf
    epoch_losses = collections.defaultdict(list)
    epoch_errors = collections.defaultdict(list)

    if model_id is None:
        model_id = str(int(time.time()))
        save_path = os.path.join(model_dir, model_id)
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = os.path.join(model_dir, model_id)
        checkpoint = torch.load(os.path.join(save_path, model_id + ".pt"),
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        last_epoch = checkpoint["last_epoch"]
        best_val_err = checkpoint["best_val_err"]
        epoch_losses = checkpoint["epoch_losses"]
        epoch_errors = checkpoint["epoch_errors"]

    best_model_wts = copy.deepcopy(model.state_dict())
    best_opt_params = copy.deepcopy(optimiser.state_dict())
    scheduler.last_epoch = last_epoch - 1
    writer = SummaryWriter(log_dir)

    since = time.time()

    num_iter = 0
    prev_conv1_weight = copy.deepcopy(model.conv1[0].weight)
    prev_conv2_weight = copy.deepcopy(model.conv2[0].weight)
    prev_conv3_weight = copy.deepcopy(model.conv3[0].weight)
    prev_conv4_weight = copy.deepcopy(model.conv4[0].weight)
    # prev_fc_weight = copy.deepcopy(model.fc.weight)
    # prev_fc_bias = copy.deepcopy(model.fc.bias)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + last_epoch, last_epoch + num_epochs - 1))
        print('-' * 10)

        model.train()
        running_loss = 0.0
        loss_total = 0
        for triplet_batch, triplet_labels in tqdm.tqdm(triplet_dataloaders["train"], desc="Triplet Batches"):
            triplet_batch = torch.reshape(triplet_batch, (-1, 3, *triplet_batch.size()[1:]))
            anchors, positives, negatives = \
                (torch.squeeze(x, dim=1) for x in torch.chunk(triplet_batch, 3, dim=1))
            # triplet_labels = torch.reshape(triplet_labels, (-1, 3))
            # anc_labels, pos_labels, neg_labels = \
            #     (torch.squeeze(x, dim=1) for x in torch.chunk(triplet_labels, 3, dim=1))

            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)

            loss, *_, conv3_out, conv2_out, conv1_out = model_forward(model, criterion, anchors, positives, negatives)
            # writer.add_embedding(anc_out, label_img=anchors, global_step=epoch, tag="anc_out")
            # writer.add_embedding(pos_out, label_img=positives, global_step=epoch, tag="pos_out")
            # writer.add_embedding(neg_out, label_img=negatives, global_step=epoch, tag="neg_out")

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            writer.add_histogram("conv1_out", conv1_out, num_iter)
            writer.add_histogram("conv2_out", conv2_out, num_iter)
            writer.add_histogram("conv3_out", conv3_out, num_iter)
            # writer.add_histogram("conv4_out", conv4_out, num_iter)

            writer.add_histogram("conv1_weight", model.conv1[0].weight, num_iter)
            writer.add_histogram("conv1_weight_diff", model.conv1[0].weight - prev_conv1_weight, num_iter)
            writer.add_histogram("conv1_grad", model.conv1[0].weight.grad, num_iter)

            writer.add_histogram("conv2_weight", model.conv2[0].weight, num_iter)
            writer.add_histogram("conv2_weight_diff", model.conv2[0].weight - prev_conv2_weight, num_iter)
            writer.add_histogram("conv2_grad", model.conv2[0].weight.grad, num_iter)

            writer.add_histogram("conv3_weight", model.conv3[0].weight, num_iter)
            writer.add_histogram("conv3_weight_diff", model.conv3[0].weight - prev_conv3_weight, num_iter)
            writer.add_histogram("conv3_grad", model.conv3[0].weight.grad, num_iter)

            writer.add_histogram("conv4_weight", model.conv4[0].weight, num_iter)
            writer.add_histogram("conv4_weight_diff", model.conv4[0].weight - prev_conv4_weight, num_iter)
            writer.add_histogram("conv4_grad", model.conv4[0].weight.grad, num_iter)

            # writer.add_histogram("fc_weight", model.fc.weight, num_iter)
            # writer.add_histogram("fc_weight_diff", model.fc.weight - prev_fc_weight, num_iter)
            # writer.add_histogram("fc_weight_grad", model.fc.weight.grad, num_iter)
            #
            # writer.add_histogram("fc_bias", model.fc.bias, num_iter)
            # writer.add_histogram("fc_bias_diff", model.fc.bias - prev_fc_bias, num_iter)
            # writer.add_histogram("fc_bias_grad", model.fc.bias.grad, num_iter)

            prev_conv1_weight = copy.deepcopy(model.conv1[0].weight)
            prev_conv2_weight = copy.deepcopy(model.conv2[0].weight)
            prev_conv3_weight = copy.deepcopy(model.conv3[0].weight)
            prev_conv4_weight = copy.deepcopy(model.conv4[0].weight)
            # prev_fc_weight = copy.deepcopy(model.fc.weight)
            # prev_fc_bias = copy.deepcopy(model.fc.bias)
            num_iter += 1

            running_loss += loss.item() * anchors.size(0)
            loss_total += anchors.size(0)

        epoch_loss = running_loss / loss_total
        print("Train Loss: {:.4f}".format(epoch_loss))
        epoch_losses["train"].append(epoch_loss)
        writer.add_scalar("train_loss", epoch_loss, epoch)

        model.eval()
        running_err = 0.0
        err_total = 0
        for pair_batch, pair_labels in pair_dataloaders["val"]:
            pair_batch = torch.reshape(pair_batch, (-1, 2, *pair_batch.size()[1:]))
            train_batch, test_batch = (torch.squeeze(x, dim=1) for x in torch.chunk(pair_batch, 2, dim=1))
            # pair_labels = torch.reshape(pair_labels, (-1, 2))
            # train_labels, test_labels = (torch.squeeze(x, dim=1) for x in torch.chunk(pair_labels, 2, dim=1))
            err = eval_forward(device, model, train_batch, test_batch)
            running_err += err
            err_total += 1

        epoch_err = running_err / err_total
        print("Val Error: {:.4f}".format(epoch_err))
        epoch_errors["val"].append(epoch_err)
        writer.add_scalar("val_error", epoch_err, epoch)

        if epoch_err < best_val_err:
            best_val_err = epoch_err
            best_model_wts = copy.deepcopy(model.state_dict())
            best_opt_params = copy.deepcopy(optimiser.state_dict())

        err = eval.evaluate_all(device, model, eval_forward, prefix="all_runs")
        print("Eval Error: {:.4f}".format(err))
        writer.add_scalar("eval_error", err, epoch)
        # scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Error: {:4f}'.format(best_val_err))

    model.load_state_dict(best_model_wts)
    optimiser.load_state_dict(best_opt_params)
    checkpoint = {
        "last_epoch": last_epoch + num_epochs,
        "best_val_err": best_val_err,
        "epoch_losses": epoch_losses,
        "epoch_errors": epoch_errors,
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict()}
    torch.save(checkpoint, os.path.join(save_path, model_id + ".pt"))
    writer.close()
    return model, model_id, checkpoint
