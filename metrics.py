import numpy as np
import torch
import torchmetrics
import torchmetrics as tm


class Metrics:
    # results should contain a sublist of [pred, labels]
    def __init__(self, num_labels):
        self.preds = []
        self.labels = []
        self.num_labels = num_labels

    def add(self, pred, label):
        self.preds.extend(pred)
        self.labels.extend(label)

    def get_reslts(self):
        return self.preds, self.labels

    def get_acc(self):
        assert len(self.preds) == len(self.labels)
        return (torch.tensor(self.preds) == torch.tensor(self.labels)).float().mean()

    def get_precision(self, mode):
        assert len(self.preds) == len(self.labels)
        precision = torchmetrics.Precision(average=mode, num_classes=self.num_labels, task="multiclass")
        return precision(torch.tensor(self.preds), torch.tensor(self.labels))

    def get_recall(self, mode):
        assert len(self.preds) == len(self.labels)
        recall = torchmetrics.Recall(average=mode, num_classes=self.num_labels, task="multiclass")
        return recall(torch.tensor(self.preds), torch.tensor(self.labels))

    def get_f1(self, mode):
        assert len(self.preds) == len(self.labels)
        macro_f1 = torchmetrics.F1Score(average=mode, num_classes=self.num_labels, task="multiclass")
        return macro_f1(torch.tensor(self.preds), torch.tensor(self.labels))


if __name__ == '__main__':
    metrics = Metrics()
    for i in range(10):
        metrics.add(np.random.randint(1, 5), np.random.randint(1, 5))
    print(metrics.get_reslts())
    print(metrics.get_acc())
