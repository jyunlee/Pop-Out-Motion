import os
import random
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset


def euclidian_distance(a, b):
    return np.sqrt(np.sum((a-b)**2, axis=1))


def k_neighbors(X_train, X_test, k=32):

    dist = [] 
    neigh_ind = []

    point_dist = [euclidian_distance(x_test, X_train) for x_test in X_test]

    for row in point_dist:
        enum_neigh = enumerate(row)
        sorted_neigh = sorted(enum_neigh, key=lambda x: x[1])[:k]

        ind_list = [tup[0] for tup in sorted_neigh]
        dist_list = [tup[1] for tup in sorted_neigh]

        dist.append(dist_list)
        neigh_ind.append(ind_list)

    return np.squeeze(np.array(neigh_ind))


def load_l_txt(file_path, n):

    f = open(file_path, 'r')

    l = np.zeros((n, n))
    
    lines = f.readlines()

    for line in lines:
        row, col, val = line.split(' ')
        l[int(row), int(col)] = float(val)

    f.close()

    return l


class DFAUSTDataset(Dataset):
    def __init__(self, split='train', data_root='../../data', npoints=4096, k=32):

        super().__init__()

        self.data_root = data_root
        self.split = split
        self.npoints = npoints
        self.k = k
        
        self.file_list = os.listdir(os.path.join(data_root, split, 'V'))
        self.file_list = [item for item in self.file_list if item.endswith('txt')]

        self.inputs, self.l_targets, self.minv_targets, self.knn = [], [], [], []

        for file in tqdm(self.file_list, total=len(self.file_list)):
            self.inputs.append(os.path.join(self.data_root, self.split, 'V', file))
            self.l_targets.append(os.path.join(self.data_root, self.split, 'L', file))
            self.minv_targets.append(os.path.join(self.data_root, self.split, 'Minv', file))
            self.knn.append(os.path.join(self.data_root, self.split, 'KNN', file[:-4] + '.npy'))


    def __getitem__(self, idx):

        if self.split != 'test':
            shape_idx, vertex_idx = idx // self.npoints, idx % self.npoints

            input = np.loadtxt(self.inputs[shape_idx])
            l_target = load_l_txt(self.l_targets[shape_idx], input.shape[0])
            minv_target = np.loadtxt(self.minv_targets[shape_idx])

            choice = np.random.choice(len(input), self.npoints, replace=False)
            choice = np.sort(choice, axis=None)

            input = input[choice, :]

            l_target = l_target[choice, choice[vertex_idx]] * 100
            minv_target = minv_target[choice[vertex_idx]] / 45952

            knn = k_neighbors(input, np.expand_dims(input[vertex_idx], axis=0), k = self.k)[1:]

            return input, vertex_idx, l_target, minv_target, knn

        else:
            shape_idx = idx

            input = np.loadtxt(self.inputs[shape_idx])
            knn = np.load(self.knn[shape_idx])
            f_name = self.inputs[shape_idx].split('/')[-1][:-4] 

            return input, knn, f_name 


    def __len__(self):

        if self.split != 'test':
            return len(self.inputs) * self.npoints
        else:
            return len(self.inputs)


if __name__ == '__main__':
    dataset = DFAUSTDataset(split='test')
    print(dataset[0])
