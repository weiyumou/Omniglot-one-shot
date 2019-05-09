import models
import eval
import torch
import numpy as np

if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    model = models.TripletNet()
    avg_acc = eval.evaluate_all(model, prefix="all_runs")
    print("Average Accuracy: {:.4f}".format(avg_acc))
