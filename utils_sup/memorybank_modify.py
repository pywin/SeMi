import torch
import torch.nn as nn
import time
import logging
import numpy as np

logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler('training_v2.log'),
        # logging.StreamHandler()
    ])
print = logging.info

class MemoryBank(nn.Module):
    def __init__(self, classes=10, num_feature_per_class=64, feature_dim=128, use_cuda=True):

        super(MemoryBank, self).__init__()
        self.classes = classes
        self.num_feature_per_class = num_feature_per_class
        self.feature_dim = feature_dim
        self.exceed_confidences = [[] for _ in range(self.classes)]
        self.beta = 0.99
        self.use_cuda = use_cuda
        if use_cuda:
            self.register_buffer('memory',torch.zeros((self.classes, self.num_feature_per_class, self.feature_dim,)).cuda())
            self.register_buffer('confidences',torch.zeros((self.classes, self.num_feature_per_class)).cuda())
        else:
            self.register_buffer('memory',torch.zeros((self.classes, self.num_feature_per_class, self.feature_dim,)))
            self.register_buffer('confidences',torch.zeros((self.classes, self.num_feature_per_class)))
    def reset_em(self):
        self.exceed_confidences = [[] for _ in range(self.classes)]

    def decay(self):
        ### V1
        for class_index in range(self.classes):
            start_a = time.time()

            # unique_elements1, count_tensor1 = torch.unique(self.confidences[class_index], return_counts=True)
            # unique_elements2, count_tensor2 = torch.unique(torch.tensor(self.exceed_confidences[class_index]).cuda(), return_counts=True)
            # intersection = unique_elements1[torch.isin(unique_elements1, unique_elements2)] #SIZE == 7
            # diff_indices = torch.nonzero(~torch.isin(self.confidences[class_index], intersection)).flatten()
            # insec1 = count_tensor1[torch.isin(unique_elements1, unique_elements2)]
            # insec2 = count_tensor2[torch.isin(unique_elements2, unique_elements1)]
            # old_elements = intersection[insec1 > insec2]
            # old_counts = insec1[insec1 > insec2] - insec2[insec1 > insec2]

            intersection = torch.from_numpy(np.intersect1d(self.confidences[class_index].cpu().numpy(), np.asarray(self.exceed_confidences[class_index]))).cuda()
            diff_indices = torch.where(~torch.isin(self.confidences[class_index], intersection))[0]
            unique_elements1, count_tensor1 = torch.unique(self.confidences[class_index], return_counts=True)
            unique_elements2, count_tensor2 = torch.unique(torch.tensor(self.exceed_confidences[class_index]).cuda(), return_counts=True)
            common_elements = torch.from_numpy(np.intersect1d(unique_elements1.cpu().numpy(), unique_elements2.cpu().numpy())).cuda()
            _, idx1 = torch.where((unique_elements1[:, None] == common_elements).T)
            _, idx2 = torch.where((unique_elements2[:, None] == common_elements).T)
            insec1 = count_tensor1[idx1]
            insec2 = count_tensor2[idx2]
            mask = insec1 > insec2
            old_elements = common_elements[mask]
            old_counts = insec1[mask] - insec2[mask]

            start_b = time.time()            
            old_indices_list = [torch.where(self.confidences[class_index] == elem)[0][:old_counts[i].item()] for i, elem in enumerate(old_elements) if len(torch.where(self.confidences[class_index] == elem)[0]) > 0]

            start_c = time.time()
            if old_indices_list:
                old_indices = torch.cat(old_indices_list, dim=0)
                difference_indices = torch.cat([old_indices.cuda(), diff_indices]).long()
            else:
                difference_indices = diff_indices

            if difference_indices.size(0) > 0:
                self.confidences[class_index][difference_indices] = self.confidences[class_index][difference_indices] * self.beta
                sorted_indices = torch.argsort(self.confidences[class_index], descending=True)
                self.memory[class_index] = self.memory[class_index][sorted_indices]
                self.confidences[class_index] = self.confidences[class_index][sorted_indices]
            
        return self.memory
        ## END
                

        ### V2
        # unique_elements1, count_tensor1 = torch.unique(self.confidences, return_counts=True)
        # unique_elements2, count_tensor2 = torch.unique(torch.tensor(self.exceed_confidences).cuda(), return_counts=True)
        # intersection = unique_elements1[torch.isin(unique_elements1, unique_elements2)]
        # diff_indices = torch.nonzero(~torch.isin(self.confidences, intersection)).flatten()
        # insec1 = count_tensor1[torch.isin(unique_elements1, unique_elements2)]
        # insec2 = count_tensor2[torch.isin(unique_elements2, unique_elements1)]
        # old_elements = intersection[insec1 > insec2]
        # old_counts = insec1[insec1 > insec2] - insec2[insec1 > insec2]

        # old_indices = []

        # for i, elem in enumerate(old_elements):
        #     index = torch.where(self.confidences == elem)[0]
        #     if len(index) > 0:
        #         old_indices.extend(index[:old_counts[i].item()])
        # difference_indices = torch.cat([torch.tensor(old_indices).cuda(), diff_indices]).long()

        # if difference_indices.size(0)>0:
        #     self.confidences[difference_indices] = self.confidences[difference_indices]*self.beta
        #     sorted_indices = torch.argsort(self.confidences, descending=True)
        #     self.memory = self.memory[sorted_indices]
        #     self.confidences = self.confidences[sorted_indices]
        # return self.memory
        ### END
        

    def push(self, batch_features, batch_targets, batch_confidences, selected_mask=None):
        start_d = time.time()
        # 添加新的特征和置信度
        batch_targets = batch_targets.long()
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

        greater_indices = selected_confidences > self.confidences[selected_targets, -1]

        updated_targets = selected_targets[greater_indices]
        updated_features = selected_features[greater_indices]
        updated_confidences = selected_confidences[greater_indices]

        self.memory[updated_targets] = torch.cat((self.memory[updated_targets, 1:], updated_features.squeeze().unsqueeze(1)), dim=1)
        self.confidences[updated_targets, -1] = updated_confidences
        #print(f"d cost time: {time.time()-start_d}")

        start_e = time.time() 
        sorted_indices = torch.argsort(self.confidences, dim=-1, descending=True)

        self.memory = torch.gather(self.memory, 1, sorted_indices.unsqueeze(2).expand(-1,-1,self.memory.size(2)))
        self.confidences = torch.gather(self.confidences, -1, sorted_indices)
        #print(f"e cost time: {time.time()-start_e}")

        start_f = time.time()
        for class_index, confidence in zip(updated_targets, updated_confidences):
            self.exceed_confidences[class_index].append(confidence.item())
        #print(f"f cost time: {time.time()-start_f}")

        return self.memory

    def get(self, class_probs, num_samples):
        start_g = time.time()
        assert len(class_probs) == self.classes
        sampled_indices = torch.multinomial(class_probs, num_samples, replacement=True)
        class_counts = torch.bincount(sampled_indices, minlength=self.classes)
        #print(f"g cost time: {time.time()-start_g}")

        start_h = time.time()
        selected_features = [self.memory[class_idx, torch.multinomial(torch.ones(self.memory.size(1)), class_counts[class_idx], replacement=False), :]
                            for class_idx, count in enumerate(class_counts) if count > 0]

        supple_features = torch.cat(selected_features, dim=0)
        supple_labels = torch.eye(self.classes)[sampled_indices]
        #print(f"h cost time: {time.time()-start_h}")
        
        return supple_features, supple_labels

    def get_protos(self):
        return torch.mean(self.memory, dim=1)

    def print_confidences(self):
        #print("Memory Bank Features:", self.memory)
        print("Memory Bank Confidences:", self.confidences)
        print("")
