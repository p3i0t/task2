import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW, Adam
from models import FeatureNet, MetricNet
from utils import AverageMeter


def ReadPairs(filename):
    """Read pairs and match labels from the given file.
    """
    pairs = []
    labels = []
    with open(filename) as f:
        for line in f:
            parts = [p.strip() for p in line.split()]
            pairs.append((parts[0], parts[3]))
            labels.append(1 if parts[1] == parts[4] else 0)

    return pairs, labels


class ComposedModel(nn.Module):
    def __init__(self):
        self.feature_net = FeatureNet()
        self.metric_net = MetricNet()

    def forward(self, left, right):
        left = self.feature_net(left)
        right = self.feature_net(right)
        lr = torch.cat([left, right], dim=1)
        logits = self.metric_net(lr)
        return logits


class CustomDataset(Dataset):
    def __init__(self, pairs, labels, dataset='yosemite'):
        super().__init__()
        self.pairs = pairs
        self.labels = labels
        self.images = np.load('{}.npz'.format(dataset))['img']

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        label = self.labels[idx]
        return self.images[pair[0]], self.images[pair[1]], label


def run_epoch(model, dataloader, optimizer=None):
    if optimizer is None:
        model.eval()  # evaluation mode
    else:
        model.train()  # train mode

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    for idx, (left, right, label) in enumerate(dataloader):
        left = left.cuda()
        right = right.cuda()
        label = label.cuda()

        score = model(left, right)
        if optimizer:
            optimizer.zero_grad()
        loss = F.cross_entropy(score, labels)

        if optimizer:
            loss.backward()
            optimizer.step()

        loss_meter.update(loss.cpu().item(), left.size(0))
        acc = (score.argmax(dim=1) == label).float().cpu().item()
        acc_meter.update(acc, left.size(0))

    return loss_meter.avg, acc_meter.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Train a MatchNet.')
    parser.add_argument("--train_set", type=str,
                        default='liberty', help="data directory.")
    parser.add_argument("--test_set", type=str,
                        default='notredame', help="data directory.")
    parser.add_argument("--n_samples", type=int,
                        default=100000, help="number of samples for training and testing.")
    parser.add_argument("--n_epochs", type=int,
                        default=100, help="number of training epochs.")

    # Optimization hyperparams:
    parser.add_argument("--batch_size", type=int,
                        default=100, help="Minibatch size")
    parser.add_argument("--optimizer", type=str,
                        default="adaw", help="adam or adamw")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Base learning rate")
    args = parser.parse_args()

    # prepare train set
    pair_filename = os.path.join(args.train_set, 'm50_{}_{}_0.txt'.format(args.n_samples, args.n_samples))
    pairs, labels = ReadPairs(pair_filename)
    train_set = CustomDataset(pairs, labels, args.train_set)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True)

    # prepare test set
    pair_filename = os.path.join(args.test_set, 'm50_{}_{}_0.txt'.format(args.n_samples, args.n_samples))
    pairs, labels = ReadPairs(pair_filename)
    test_set = CustomDataset(pairs, labels, args.test_set)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=8, pin_memory=True, drop_last=True)

    model = ComposedModel().cuda()

    optimizer = AdamW(model.parameters(), lr=args.lr)

    optimal_loss = 1e5
    for epoch in range(args.n_epochs):
        print('====> Epoch {}'.format(epoch + 1))
        train_loss, train_acc = run_epoch(model, train_loader, optimizer=optimizer)
        print('train loss: {:.4f}, train acc: {:.4f}'.format(train_loss, train_acc))

        test_loss, test_acc = run_epoch(model, test_loader)
        print('test loss: {:.4f}, test acc: {:.4f}'.format(test_loss, test_acc))

        if train_loss < optimal_loss:
            print('===> new optimal found.')
            optimal_loss = train_loss
            torch.save(model.state_dict(), 'model.pt')
