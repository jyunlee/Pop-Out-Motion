import os
import argparse
import numpy as np
import pymeshlab


def parse_args():
    parser = argparse.ArgumentParser('Model') 
    parser.add_argument('--in_dir', type=str, default='../in_data', help='path to input data directory')
    parser.add_argument('--out_dir', type=str, default='../data/test', help='path to output data directory')
    parser.add_argument('--k', type=int, default=32, help='hyper-parameter k to select KNN pairs')

    return parser.parse_args() 


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


if __name__ == '__main__':

    args = parse_args()

    in_dir, out_dir, k = args.in_dir, args.out_dir, args.k

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for sub_dir in ['Mesh', 'V', 'KNN']:
        if not os.path.exists(os.path.join(out_dir, sub_dir)):
            os.mkdir(os.path.join(out_dir, sub_dir))


    for file in os.listdir(in_dir):

        if not file.endswith('.obj'):
            continue

        in_file = os.path.join(in_dir, file) 
        out_file = os.path.join(out_dir, 'Mesh', file)
    
        # normalize
        os.system('%s %s %s' % ('./build/normalize_bin', in_file, out_file))

        # save V
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(out_file)

        v = ms.current_mesh().vertex_matrix()

        np.savetxt(os.path.join(out_dir, 'V', file[:-4] + '.txt'), v)

        # calculate KNN pairs
        knn_list = np.zeros((v.shape[0], k-1)) 

        for i in range(v.shape[0]):
            knn = k_neighbors(v, np.expand_dims(v[i], axis=0), k=k)[1:]
            knn_list[i] = knn

        np.save(os.path.join(out_dir, 'KNN', file[:-4] + '.npy'), knn_list)


