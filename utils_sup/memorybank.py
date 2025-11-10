import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class MemoryBank(nn.Module):
    def __init__(self, classes=10, num_feature_per_class=64, feature_dim=128, use_cuda=True, device=None):
        super(MemoryBank, self).__init__()
        self.classes = classes
        self.num_feature_per_class = num_feature_per_class
        self.feature_dim = feature_dim
        self.exceed_confidences = [[] for _ in range(self.classes)]
        self.beta = 0.99
        self.use_cuda = use_cuda
        self.device = device
        if use_cuda:
            self.register_buffer('memory',torch.zeros((self.classes, self.num_feature_per_class, self.feature_dim,)).to(self.device))
            self.register_buffer('confidences',torch.zeros((self.classes, self.num_feature_per_class)).to(self.device))
        else:
            self.register_buffer('memory',torch.zeros((self.classes, self.num_feature_per_class, self.feature_dim,)))
            self.register_buffer('confidences',torch.zeros((self.classes, self.num_feature_per_class)))
    def reset_em(self):
        self.exceed_confidences = [[] for _ in range(self.classes)]

    def decay(self):
        #print('*'*100)
        #print(f'exceed:{self.exceed_confidences[1]}')
        #print(f'exceed_len:{len(self.exceed_confidences[1])}')
        for class_index in range(self.classes):
            #if class_index ==1:
            #    print(f'class_1:{self.confidences[class_index]}')
            unique_elements1, count_tensor1 = torch.unique(self.confidences[class_index], return_counts=True)
            unique_elements2, count_tensor2 = torch.unique(torch.tensor(self.exceed_confidences[class_index]).to(self.device), return_counts=True)
            intersection = unique_elements1[torch.isin(unique_elements1, unique_elements2)]
            #diff = self.confidences[class_index][~torch.isin(self.confidences[class_index], intersection)]
            diff_indices = torch.nonzero(~torch.isin(self.confidences[class_index], intersection)).flatten()
            insec1 = count_tensor1[torch.isin(unique_elements1, unique_elements2)]
            insec2 = count_tensor2[torch.isin(unique_elements2, unique_elements1)]
            # 找到在tensor2中但不在tensor1中的元素
            old_elements = intersection[insec1 > insec2]
            old_counts = insec1[insec1 > insec2] - insec2[insec1 > insec2]

            old_indices = []

            for i, elem in enumerate(old_elements):
                # 使用 torch.where 找到元素在 tensor1 中的索引
                index = torch.where(self.confidences[class_index] == elem)[0]
                # 如果找到了索引，将第一个索引添加到列表中
                if len(index) > 0:
                    old_indices.extend(index[:old_counts[i].item()])
            difference_indices = torch.cat([torch.tensor(old_indices).to(self.device), diff_indices]).long()

            if difference_indices.size(0)>0:
                #if class_index==1:
                #    print(f'diff_idxs:{difference_indices}')
                #    print(f'diff_idxs_len:{difference_indices.size(0)}')
                self.confidences[class_index][difference_indices] = self.confidences[class_index][difference_indices]*self.beta
                sorted_indices = torch.argsort(self.confidences[class_index], descending=True)
                self.memory[class_index] = self.memory[class_index][sorted_indices]
                self.confidences[class_index] = self.confidences[class_index][sorted_indices]
                #if class_index==1:
                #    print(f'class_1:{self.confidences[class_index]}')
        #print('*'*100)

        return  self.memory

    def push(self, batch_features, batch_targets, batch_confidences, selected_mask=None):
        # 添加新的特征和置信度
        #batch_targets = batch_targets.long()
        #print(batch_targets)
        #batch_confidences = batch_confidences.long()
        #print(f'batch_confidences:{batch_confidences}')
        if selected_mask is not None and not torch.all(selected_mask==0):
            selected_mask = selected_mask.bool()
            selected_features = batch_features[selected_mask]
            selected_targets = batch_targets[selected_mask]
            selected_confidences = batch_confidences[selected_mask]
            selected_targets = torch.argmax(selected_targets, dim=1)
            #print(f'selected_confidences:{selected_confidences}')
            #print(f'selected_targets:{selected_targets}')
        else:
            return self.memory

        for i in range(selected_features.size(0)):
            # 获取样本标签
            class_index = selected_targets[i]
            # 获取样本对应的confidence
            selected_confidence = selected_confidences[i]
            # 获取样本对应的feature
            selected_feature = selected_features[i].squeeze().unsqueeze(0)
            # 满足条件，排序并更新Memory Bank
            #print(selected_confidence)
            #print(self.confidences[class_index][-1])
            if selected_confidence > self.confidences[class_index][-1]:
                #if class_index ==1:
                    #print(f'class:{class_index} -- {self.confidences[class_index]}')
                #self.memory[class_index][-1] = selected_feature.squeeze()
                self.memory[class_index] = torch.cat((self.memory[class_index][1:], selected_feature))
                self.confidences[class_index][-1] = selected_confidence
                #if class_index == 1:
                    #print(f'class:{class_index} -- {self.confidences[class_index]}')
                # _, sorted_indices = torch.sort()
                sorted_indices = torch.argsort(self.confidences[class_index], descending=True)
                self.exceed_confidences[class_index].append(selected_confidence)
                #if class_index == 1:
                    #print(f'class:{class_index} -- {self.exceed_confidences[class_index]}')
                self.memory[class_index] = self.memory[class_index][sorted_indices]
                self.confidences[class_index] = self.confidences[class_index][sorted_indices]
                #if class_index == 1:
                    #print(f'class:{class_index} -- {self.confidences[class_index]}')
        return self.memory

    def get(self, class_probs, num_samples):
        assert len(class_probs) == self.classes
        # 从每个 class 中按照概率采样特征
        selected_features = []
        selected_labels = []
        sampled_indices = torch.multinomial(class_probs, num_samples, replacement=True)
        # 计算每个类别采样了多少个样本
        class_counts = torch.bincount(sampled_indices, minlength=self.classes)
        for class_idx in range(self.memory.size(0)):
            class_num_samples = class_counts[class_idx]
            if class_num_samples == 0:
                continue
            # 从当前 class 中采样 num_samples 个特征
            sampled_indices = torch.multinomial(torch.ones(self.memory.size(1)), class_num_samples, replacement=False)
            class_features = self.memory[class_idx, sampled_indices, :]
            selected_features.append(class_features)
            selected_labels.extend([class_idx] * class_num_samples)
        # 合并成一个二维张量
        supple_features = torch.cat(selected_features, dim=0)
        supple_labels = torch.tensor(selected_labels)
        supple_labels = torch.eye(self.classes)[supple_labels]
        return supple_features, supple_labels

    def get_sup(self, labeled_target, unlabeled_target, selected_mask, num_samples=128):
        unlabeled_target = unlabeled_target[selected_mask.bool()]
        labels = torch.cat((labeled_target, unlabeled_target), dim=0)
        labels = torch.argmax(labels, dim=1)
        total_samples = torch.tensor(labels.size(0) + num_samples).to(self.device)
        labeled_class_counts = torch.zeros(self.classes).to(self.device)
        labeled_class_counts.scatter_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
        base_count = torch.div(total_samples, self.classes, rounding_mode='trunc')
        exceed_base = (labeled_class_counts >=base_count)
        false_count = torch.sum(exceed_base == False)
        allocable = total_samples - labeled_class_counts[exceed_base].sum()
        base_count = torch.div(allocable, false_count, rounding_mode='trunc')
        remainder = torch.remainder(allocable, base_count)
        true_indices = [i for i, val in enumerate(exceed_base) if not val]
        random_indices = random.sample(true_indices, int(remainder))
        target_counts = torch.ones(self.classes).to(self.device)*base_count
        for idx in random_indices:
            target_counts[idx] += 1
        class_counts = [max(0, target_counts[i] - count) for i, count in enumerate(labeled_class_counts)]
        class_counts_sum = int(sum(class_counts).item())
        supple_features = torch.zeros(class_counts_sum, self.memory.size(2)).to(self.device)
        supple_labels = torch.zeros(class_counts_sum).to(self.device)
        n = 0
        for class_idx in range(self.memory.size(0)):
            class_num_samples = int(class_counts[class_idx])
            if class_num_samples == 0:
                continue
            # 从当前 class 中采样 num_samples 个特征
            #sampled_indices = torch.multinomial(torch.ones(self.memory.size(1)), class_num_samples, replacement=False)
            sampled_indices = torch.randperm(self.memory.size(1))[:class_num_samples]
            class_features = self.memory[class_idx, sampled_indices, :]
            supple_features[n:n+class_num_samples] = class_features
            supple_labels[n:n+class_num_samples] = torch.tensor([class_idx] * class_num_samples).to(self.device)
            n += class_num_samples
        # 合并成一个二维张量
        supple_labels = torch.eye(self.classes)[supple_labels.long()]
        return supple_features, supple_labels

    def get_current_memory(self):
        return self.memory

    def get_protos(self):
        return torch.mean(self.memory, dim=1)

    def print_confidences(self):
        #print("Memory Bank Features:", self.memory)
        print("Memory Bank Confidences:", self.confidences)
        print("")
