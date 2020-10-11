import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW, Adam
from models import FeatureNet, MetricNet
from utils import AverageMeter, ErrorRateAt95Recall


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
        super().__init__()

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
        return self.images[int(pair[0])], self.images[int(pair[1])], label


def preprocess(x):
    return (x.float() - 128) / 160  # from original paper.


def run_epoch(model, dataloader, optimizer=None):
    if optimizer is None:
        model.eval()  # evaluation mode
    else:
        model.train()  # train mode

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for idx, (left, right, label) in enumerate(dataloader):
        left = preprocess(left).cuda()
        right = preprocess(right).cuda()
        label = label.cuda()

        score = model(left, right)
        if optimizer:
            optimizer.zero_grad()
        loss = F.cross_entropy(score, label)

        if optimizer:
            loss.backward()
            optimizer.step()

        loss_meter.update(loss.cpu().item(), left.size(0))
        acc = (score.argmax(dim=1) == label).float().cpu().mean().item()
        acc_meter.update(acc, left.size(0))

    return loss_meter.avg, acc_meter.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Train a MatchNet.')
    parser.add_argument("--eval", action="store_true",
                        help="Use in eval mode")
    parser.add_argument("--train_set", type=str,
                        default='liberty', help="train data directory.")
    parser.add_argument("--valid_set", type=str,
                        default='yosimite', help="valid data directory.")
    parser.add_argument("--test_set", type=str,
                        default='notredame', help="data directory.")
    parser.add_argument("--n_samples_train", type=int,
                        default=500000, help="number of samples for training.")
    parser.add_argument("--n_samples_test", type=int,
                        default=100000, help="number of samples for testing.")
    parser.add_argument("--n_epochs", type=int,
                        default=100, help="number of training epochs.")

    # Optimization hyperparams:
    parser.add_argument("--batch_size", type=int,
                        default=32, help="Minibatch size")
    parser.add_argument("--optimizer", type=str,
                        default="adaw", help="adam or adamw")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Base learning rate")
    args = parser.parse_args()

    # prepare train set
    pair_filename = os.path.join(args.train_set, 'm50_{}_{}_0.txt'.format(args.n_samples_train, args.n_samples_train))
    pairs, labels = ReadPairs(pair_filename)
    train_set = CustomDataset(pairs, labels, args.train_set)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True)
    
    pair_filename = os.path.join(args.valid_set, 'm50_{}_{}_0.txt'.format(args.n_samples_test, args.n_samples_test))
    pairs, labels = ReadPairs(pair_filename)
    valid_set = CustomDataset(pairs, labels, args.valid_set)
    valid_loader = DataLoader(valid_set, batch_size=100, shuffle=False,
                              num_workers=8, pin_memory=True, drop_last=True)

    # prepare test set
    pair_filename = os.path.join(args.test_set, 'm50_{}_{}_0.txt'.format(args.n_samples_test, args.n_samples_test))
    pairs, labels = ReadPairs(pair_filename)
    test_set = CustomDataset(pairs, labels, args.test_set)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False,
                             num_workers=8, pin_memory=True, drop_last=True)

    model = ComposedModel().cuda()
    if args.eval:
        state_dict = torch.load('model_{}.pt'.format(args.train_set))
        model.load_state_dict(state_dict)
        model.eval()

        score_list = []
        label_list = []
        for idx, (left, right, label) in enumerate(test_loader):
            left = preprocess(left).cuda()
            right = preprocess(right).cuda()
            label = label.cuda()

            score = model(left, right)
            score = F.softmax(score, dim=-1)[:, 1]
            score_list += list(score.cpu().detach().numpy())
            label_list += list(label.cpu().numpy())
        print(score_list)
        err_rate = ErrorRateAt95Recall(labels=label_list, scores=score_list)
        print('Error Rate 95% Recall: {:.4f}'.format(err_rate))

    else:
        optimizer = AdamW(model.parameters(), lr=args.lr)

        optimal_loss = 1e5
        for epoch in range(args.n_epochs):
            print('====> Epoch {}'.format(epoch + 1))
            train_loss, train_acc = run_epoch(model, train_loader, optimizer=optimizer)
            print('train loss: {:.4f}, train acc: {:.4f}'.format(train_loss, train_acc))

            valid_loss, valid_acc = run_epoch(model, valid_loader)
            print('valid loss: {:.4f}, valid acc: {:.4f}'.format(valid_loss, valid_acc))

            test_loss, test_acc = run_epoch(model, test_loader)
            print('test loss: {:.4f}, test acc: {:.4f}'.format(test_loss, test_acc))

            if valid_loss < optimal_loss:
                print('===> new optimal found.')
                optimal_loss = valid_loss
                torch.save(model.state_dict(), 'model_{}.pt'.format(args.train_set))
