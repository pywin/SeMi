from __future__ import print_function

import os
import copy
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.cluster import AgglomerativeClustering
import random


__all__ = ['make_imb_data', 'save_checkpoint', 'update_qbar', 'MixMatch_Loss', 'FixMatch_Loss', 
           'ReMixMatch_Loss', 'linear_rampup', 'get_weighted_sampler', 'merge_two_datasets',
           'WeightEMA', 'interleave', 'CenterLoss', 'Balanced_FixMatch_Loss', 
           'SupervisedContrastiveLoss', 'ProtoClassifier', 'probs_monitor', 'probs_monitor_base',
           'get_labeled_class_weights', 'probs_monitor_', 'get_fuse_class_weights']

def probs_monitor_old(target_u, label_counts, select_mask, return_probs=False):
    target_u = torch.argmax(target_u, dim=1)
    target_u = target_u[select_mask.bool()]
    label_counts_u = torch.bincount(target_u, minlength=10)
    label_counts += label_counts_u
    if return_probs:
        label_probabilities = label_counts.sum() / label_counts.float()
        total_probability = torch.sum(label_probabilities)
        normalized_probabilities = label_probabilities / total_probability
        return normalized_probabilities

def probs_monitor(target_u, label_counts, select_mask, epoch, decay_factor=0.99, return_probs=False):
    target_u = torch.argmax(target_u, dim=1)
    target_u = target_u[select_mask.bool()]
    label_counts_u = torch.bincount(target_u, minlength=10)
    label_counts += label_counts_u
    if return_probs:
        label_probabilities = label_counts.sum() / label_counts.float()
        total_probability = torch.sum(label_probabilities)
        normalized_probabilities = label_probabilities / total_probability
        label_probabilities_inv = label_counts.float() / label_counts.sum()
        total_probability_inv = torch.sum(label_probabilities_inv)
        normalized_probabilities_inv = label_probabilities_inv / total_probability_inv
        print(f'norm probs:{normalized_probabilities}')
        log_probs = torch.log(normalized_probabilities)
        log_probs_inv = torch.log(normalized_probabilities_inv)
        print(f'log probs:{log_probs}')
        restored_probs = torch.exp(log_probs * (1 - decay_factor ** (epoch+1)))
        restored_probs_inv = torch.exp(log_probs_inv * (1 - decay_factor ** (epoch+1)))
        print(f'restored probs:{restored_probs}')
        print(f'restored probs_inv:{restored_probs_inv}')
        return restored_probs, restored_probs_inv

def probs_monitor_(target_u, label_counts, select_mask, epoch, num_class, decay_factor=0.99, return_probs=False):
    target_u = torch.argmax(target_u, dim=1)
    target_u = target_u[select_mask.bool()]
    label_counts_u = torch.bincount(target_u, minlength=num_class)
    label_counts += label_counts_u
    if return_probs:
        label_probabilities = label_counts.sum() / label_counts.float()
        total_probability = torch.sum(label_probabilities)
        normalized_probabilities = label_probabilities / total_probability
        label_probabilities_inv = label_counts.float() / label_counts.sum()
        total_probability_inv = torch.sum(label_probabilities_inv)
        normalized_probabilities_inv = label_probabilities_inv / total_probability_inv
        print(f'norm probs:{normalized_probabilities}')
        log_probs = torch.log(normalized_probabilities)
        log_probs_inv = torch.log(normalized_probabilities_inv)
        print(f'log probs:{log_probs}')
        restored_probs = torch.exp(log_probs * (1 - decay_factor ** (epoch+1)))
        restored_probs_inv = torch.exp(log_probs_inv * (1 - decay_factor ** (epoch+1)))
        print(f'restored probs:{restored_probs}')
        print(f'restored probs_inv:{restored_probs_inv}')
        return restored_probs, restored_probs_inv, normalized_probabilities_inv
    else:
        return label_counts

def probs_monitor_base(target_u, label_counts, select_mask, epoch, decay_factor=0.99, return_probs=False):
    target_u = torch.argmax(target_u, dim=1)
    target_u = target_u[select_mask.bool()]
    label_counts_u = torch.bincount(target_u, minlength=10)
    label_counts += label_counts_u
    if return_probs:
        label_probabilities = label_counts.sum() / label_counts.float()
        total_probability = torch.sum(label_probabilities)
        normalized_probabilities = label_probabilities / total_probability
        label_probabilities_inv = label_counts.float() / label_counts.sum()
        total_probability_inv = torch.sum(label_probabilities_inv)
        normalized_probabilities_inv = label_probabilities_inv / total_probability_inv
        print(f'norm probs:{normalized_probabilities}')
        log_probs = torch.log(normalized_probabilities)
        log_probs_inv = torch.log(normalized_probabilities_inv)
        print(f'log probs:{log_probs}')
        restored_probs = torch.exp(log_probs * (1 - decay_factor ** (epoch+1)))
        restored_probs_inv = torch.exp(log_probs_inv * (1 - decay_factor ** (epoch+1)))
        print(f'restored probs:{restored_probs}')
        print(f'restored probs_inv:{restored_probs_inv}')
        return restored_probs, restored_probs_inv, label_counts.float()/label_counts.sum() 

def get_labeled_class_weights(class_counts):
    class_weights = [1 / count for count in class_counts]
    total_weight = sum(class_weights)
    normalized_weights = [weight / total_weight for weight in class_weights]
    return normalized_weights

def get_fuse_class_weights(labeled_class_counts, unlabeled_class_counts):
    fuse_class_counts = labeled_class_counts + unlabeled_class_counts
    return get_labeled_class_weights(fuse_class_counts)

def make_imb_data(max_num, class_num, gamma):
    mu = np.power(1 / abs(gamma), 1 / (class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / abs(gamma)))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))
    if gamma < 0:
        class_num_list = class_num_list[::-1]
    # print(class_num_list)
    return list(class_num_list)


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

@torch.no_grad()
def update_qbar(qbar, q):
    """
    Update center used for projection.
    """
    batch_center = torch.sum(q, dim=0, keepdim=True)
    batch_center = batch_center / len(q)
    # ema update
    qbar = qbar * 0.9 + batch_center * (1 - 0.9)
    return qbar


class ReMixMatch_Loss(object):
    def __init__(self, lambda_u, rampup_length):
        self.lambda_u = lambda_u
        self.rampup_length = rampup_length

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = -torch.mean(torch.sum(F.log_softmax(outputs_u, dim=1) * targets_u, dim=1))

        return Lx, Lu, self.lambda_u * linear_rampup(epoch, self.rampup_length)


class MixMatch_Loss(object):
    def __init__(self, lambda_u, rampup_length):
        self.lambda_u = lambda_u
        self.rampup_length = rampup_length

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, self.lambda_u * linear_rampup(epoch, self.rampup_length)

class FixMatch_Loss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, mask, ohem_weights=None):
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        if ohem_weights is not None:
            Lu = -torch.mean(torch.sum(F.log_softmax(outputs_u, dim=1) * targets_u, dim=1) * ohem_weights * mask)
        else:
            Lu = -torch.mean(torch.sum(F.log_softmax(outputs_u, dim=1) * targets_u, dim=1) * mask)
        return Lx, Lu

class Balanced_FixMatch_Loss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, outputs_sup, targets_sup, mask):
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = -torch.mean(torch.sum(F.log_softmax(outputs_u, dim=1) * targets_u, dim=1) * mask)
        Lsup = -torch.mean(torch.sum(F.log_softmax(outputs_sup, dim=1) * targets_sup, dim=1))
        return Lx, Lu, Lsup


class WeightEMA(object):
    def __init__(self, model, ema_model, lr, alpha=0.999, wd=True):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        if wd:
            self.wd = 0.02 * lr
        else:
            self.wd = 0.0

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param = ema_param.float()
            param = param.float()
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param.mul_(1 - self.wd)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def save_checkpoint(state, epoch, checkpoint, filename='checkpoint.pth.tar', isBest=False):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if epoch % 100 == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, f'checkpoint_{epoch + 1}.pth.tar'))
    if isBest:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'checkpoint_best.pth.tar'))


def get_weighted_sampler(target_sample_rate, num_sample_per_class, target):
    assert len(num_sample_per_class) == len(np.unique(target))

    sample_weights = target_sample_rate / num_sample_per_class  # this is the key line!!!

    # assign each sample a weight by sampling rate
    samples_weight = np.array([sample_weights[t] for t in target])

    return WeightedRandomSampler(samples_weight, len(samples_weight), True)


class merge_two_datasets(torch.utils.data.Dataset):
    def __init__(self, data1, data2, targets1, targets2,
                 transform=None, target_transform=None):
        self.data = copy.deepcopy(data1 + data2)
        self.targets = copy.deepcopy(np.concatenate([targets1, targets2], axis=0))
        assert len(self.data) == len(self.targets)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)


class CenterLoss2(nn.Module):
    def __init__(self, num_classes=10, feat_dim=128, use_gpu=True):
        super(CenterLoss2, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels, select_mask=None):
        if select_mask is not None:
            select_mask = select_mask.long()
            x = x[select_mask]
            labels = labels[select_mask]
        labels = torch.argmax(labels, dim=1)
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

class CenterLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=128, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels, select_mask=None, class_weights=None):
        if select_mask is not None:
            select_mask = select_mask.bool()
            x = x[select_mask]
            labels = labels[select_mask]
        x = x.squeeze()
        labels = torch.argmax(labels, dim=1)
        #print(x.size())          #torch.Size([64, 128])
        #print(labels.size())     #torch.Size([64])
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        #print(distmat)
        if class_weights is not None:
            selected_weights = class_weights
            #print(selected_weights)
            distmat = distmat * selected_weights
            #print(distmat)
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, projections, targets, select_mask=None, class_weights=None):
        if select_mask is not None:
            if torch.all(select_mask==0):
                return torch.tensor(0.0, dtype=torch.float32).cuda()
            select_mask = select_mask.bool()
            projections = projections[select_mask]
            targets = targets[select_mask]
        #print(f'org_projection_size:{projections.size()}')
        projections = projections.squeeze()
        #print(f'projection_size:{projections}')
        targets = torch.argmax(targets, dim=1)
        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")
        dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
        #print(f'dot_product_tempered_size:{dot_product_tempered}')
        # Check if dot_product_tempered is empty along dimension 1
        if dot_product_tempered.size(1) == 0:
            # Handle the case where dot_product_tempered is empty
            return torch.tensor(0.0, dtype=torch.float32).cuda()
        exp_dot_tempered = (torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5)
        #print(f'exp_dot_tempered:{exp_dot_tempered}')

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)
        non_zero_cardinality_mask = cardinality_per_samples != 0
        #print(cardinality_per_samples[non_zero_cardinality_mask].size())

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        #print(f'log_prob:{log_prob.size()}')
        supervised_contrastive_loss_per_sample = torch.sum(log_prob[non_zero_cardinality_mask] * mask_combined[non_zero_cardinality_mask], dim=1) / cardinality_per_samples[non_zero_cardinality_mask]
        if class_weights is not None:
        # Apply class weights to the loss
            class_weights = class_weights.to(device)
            target_weights = torch.gather(class_weights, dim=0, index=targets)
            supervised_contrastive_loss_per_sample = supervised_contrastive_loss_per_sample * target_weights[non_zero_cardinality_mask]
        #print(f'loss_per_sample:{supervised_contrastive_loss_per_sample}')
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss

class ProtoClassifier_(nn.Module):
    def __init__(self, num_classes):
        super(ProtoClassifier_, self).__init__()
        self.num_classes = num_classes


    def ic_func(self, class_features):
        return torch.sum(torch.cdist(class_features, class_features), dim=0)

    @torch.no_grad()
    def forward(self, inp, memory, selected_center=2, T=1.0):
        selected_features = [[] for _ in range(self.num_classes)]
        proto_center = torch.zeros(self.num_classes, memory.size(2)).cuda()
        for cls in range(self.num_classes):
            results = self.ic_func(memory[cls]).cpu().reshape(-1, 1)
            #kmeans = KMeans(n_clusters=3, random_state=0, n_init=10, init='k-means++', max_iter=50).fit(results)
            #temp_label = KMeans(n_clusters=3, mode='euclidean').fit_predict(results)
            #temp_label = kmeans.labels_
            temp_label = AgglomerativeClustering(n_clusters=3).fit_predict(results)
            c0 = torch.arange(memory.size(1))[temp_label==0]
            c1 = torch.arange(memory.size(1))[temp_label==1]
            c2 = torch.arange(memory.size(1))[temp_label==2]

            # 使用 zip 将三个列表打包成元组列表，每个元组包含列表的长度和列表本身
            lists_and_lengths = list(zip([c0, c1, c2], map(len, [c0, c1, c2])))

            # 使用 sorted 对元组列表按照长度进行降序排序
            sorted_lists = sorted(lists_and_lengths, key=lambda x: x[1], reverse=True)
            # 使用循环解压排序后的列表
            for i, (lst, _) in enumerate(sorted_lists):
                if i<selected_center:
                    selected_features[cls].extend(memory[cls][lst])
            proto_center[cls] = torch.mean(torch.stack(selected_features[cls]).float(), dim=0)
        dist = torch.cdist(inp, proto_center)
        return F.softmax(-dist * T, dim=1)


class ProtoClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ProtoClassifier, self).__init__()

    @torch.no_grad()
    def forward(self, inp, proto_centers, T=1.0, norm=True):
        #inp = F.normalize(inp, dim=1)
        if norm:
            proto_centers = F.normalize(proto_centers, dim=1)
        dist = torch.cdist(inp, proto_centers)
        return F.softmax(-dist / T, dim=1)
