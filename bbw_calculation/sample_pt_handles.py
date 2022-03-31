import os
import numpy
import argparse
import pymeshlab
from chainer import cuda
import open3d as o3d


def l2_dist(x, y):
    return ((x - y) ** 2).sum(axis=2)


def farthest_point_sampling(pts, k=16, initial_idx=0, metrics=l2_dist,
                            skip_initial=True, indices_dtype=numpy.int32,
                            distances_dtype=numpy.float32):
    if pts.ndim == 2:
        # insert batch_size axis
        pts = pts[None, ...]

    assert pts.ndim == 3

    xp = cuda.get_array_module(pts)
    batch_size, num_point, coord_dim = pts.shape
    indices = xp.zeros((batch_size, k, ), dtype=indices_dtype)

    # distances[bs, i, j] is distance between i-th farthest point `pts[bs, i]`
    # and j-th input point `pts[bs, j]`.
    distances = xp.zeros((batch_size, k, num_point), dtype=distances_dtype)
    if initial_idx is None:
        indices[:, 0] = xp.random.randint(len(pts))
    else:
        indices[:, 0] = initial_idx

    batch_indices = xp.arange(batch_size)
    farthest_point = pts[batch_indices, indices[:, 0]]

    # minimum distances to the sampled farthest point
    try:
        min_distances = metrics(farthest_point[:, None, :], pts)
    except Exception as e:
        import IPython; IPython.embed()

    if skip_initial:
        # Override 0-th `indices` by the farthest point of `initial_idx`
        indices[:, 0] = xp.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, 0]]
        min_distances = metrics(farthest_point[:, None, :], pts)

    distances[:, 0, :] = min_distances

    for i in range(1, k):
        indices[:, i] = xp.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, i]]
        dist = metrics(farthest_point[:, None, :], pts)
        distances[:, i, :] = dist
        min_distances = xp.minimum(min_distances, dist)

    return indices, distances


def parse_args():
    parser = argparse.ArgumentParser('Model')

    parser.add_argument('--in_dir', type=str, default='../data', help='path to input data directory')
    parser.add_argument('--out_dir', type=str, default='./handles', help='path to output point handle directory')
    parser.add_argument('--n_handle', type=int, default=16, help='number of point handles')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    in_dir = os.path.join(args.in_dir, 'test', 'V')
    out_dir = args.out_dir

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for f in os.listdir(in_dir):

        if not f.endswith('.txt'):
            continue

        f_path = os.path.join(in_dir, f)
        v = numpy.loadtxt(f_path)

        indices, _ = farthest_point_sampling(v, k=args.n_handle) 

        pcd = v[indices, :]
        
        out_pcd = o3d.geometry.PointCloud()
        out_pcd.points = o3d.utility.Vector3dVector(pcd[0])

        out_f_path = os.path.join(out_dir, f[:-4] + '.ply')
        o3d.io.write_point_cloud(out_f_path, out_pcd)

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(out_f_path)
        ms.save_current_mesh(out_f_path[:-4] + '.obj')

        os.system('rm %s' % out_f_path)

    

