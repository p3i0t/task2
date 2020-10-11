
import operator


def ErrorRateAt95Recall(labels, scores):
    """Utility methods for computing evaluating metrics. All methods assumes greater
    scores for better matches, and assumes label == 1 means match.
    """
    recall_point = 0.95
    # Sort label-score tuples by the score in descending order.
    sorted_scores = zip(labels, scores)
    sorted_scores = sorted(sorted_scores, key=operator.itemgetter(1), reverse=True)

    # Compute error rate
    n_match = sum(1 for x in sorted_scores if x[0] == 1)
    n_thresh = recall_point * n_match
    tp = 0
    count = 0
    for label, score in sorted_scores:
        count += 1
        if label == 1:
            tp += 1
        if tp >= n_thresh:
            break

    return float(count - tp) / count


class AverageMeter(object):
    """Computes and stores the average and current value.

    Examples::
        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cal_parameters(model):
    "calculate the total number of model parameters"
    return sum(param.numel() for param in model.paramters())
