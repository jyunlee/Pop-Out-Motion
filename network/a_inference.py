import os
import argparse
import torch
import torch.nn as nn
import sys
import numpy as np
import time
import hydra
import omegaconf
import pytorch_lightning as pl

from tqdm import tqdm
from multiprocessing import Pool
from data.DFAUSTLoader import DFAUSTDataset

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='lap_net', help='model name')
    parser.add_argument('--gpu', type=str, default='0', help='GPU number')
    parser.add_argument('--data_dir', type=str, default='/workspace/Pop-Out-Motion/data/', help='path to input data directory')
    parser.add_argument('--checkpoint_path', type=str, default='/workspace/Pop-Out-Motion/network/outputs/lap_net/best_model.ckpt', help='path to model checkpoint')
    parser.add_argument('--out_dir', type=str, default='/workspace/Pop-Out-Motion/network/a_out', help='path to output directory')
    parser.add_argument('--k', type=int, default=32, help='k to select KNN pairs')
    parser.add_argument('--n_points', type=int, default=128, help='number of reference points to regress at a time')

    return parser.parse_args()

args = parse_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

def hydra_params_to_dotdict(hparams):
    def _to_dot_dict(cfg):
        res = {}
        for k, v in cfg.items():
            if isinstance(v, omegaconf.DictConfig):
                res.update(
                    {k + "." + subk: subv for subk, subv in _to_dot_dict(v).items()}
                )
            elif isinstance(v, (str, int, float, bool)):
                res[k] = v

        return res
    return _to_dot_dict(hparams)

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def postprocess_l_minv(idx):

    file_list = os.listdir(os.path.join(args.data_dir, 'test', 'V'))
    file = file_list[idx]

    l_flatten = np.load(os.path.join(args.out_dir, file[:-4]+'_l.npy')).astype(float)
    minv = np.load(os.path.join(args.out_dir, file[:-4]+'_minv.npy')).astype(float)
    knn = np.load(os.path.join(args.data_dir, 'test', 'KNN', file[:-4]+'.npy')).astype(int)

    l = np.zeros((l_flatten.shape[0], l_flatten.shape[0]))

    for point_idx in range(l.shape[0]):
        l[point_idx, knn[point_idx]]= l_flatten[point_idx]

    # un-normalize
    minv *= 45952
    minv = np.diag(minv)

    l_flatten /= 100
 
    # fix assymetric knn samplings
    for i in range(l.shape[0]):
        for j in range(l.shape[1]):
            if l[i, j] == 0:
                l[j, i] = 0

    for i in range(l.shape[0]):
        l[i, i] = -1 * (l[i].sum() - l[i, i])

    a = np.matmul(l, np.matmul(minv, l))
    
    # for sanity check
    if False:    
        print('Symmetry: ', np.all(np.abs(a-a.T) < 0.0000001))

        eig_val = np.linalg.eigvals(a)
        eig_val.sort()

        print('Eig vals: ', eig_val)
   
    # add small-valued diagonal matrix to enhance robustness of the later optimization process
    a += (np.identity(a.shape[0]) * 0.00000001)

    np.savetxt(os.path.join(args.out_dir, file[:-4]+'.txt'), a)

    os.system('rm %s' % os.path.join(args.out_dir, file[:-4]+'_l.npy'))
    os.system('rm %s' % os.path.join(args.out_dir, file[:-4]+'_minv.npy'))


@hydra.main("./config/config.yaml")
def main(cfg):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''MODEL LOADING'''
    regressor = hydra.utils.instantiate(cfg.task_model, hydra_params_to_dotdict(cfg))

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    checkpoint = torch.load(args.checkpoint_path)
    regressor.load_state_dict(checkpoint['state_dict'])
    regressor = regressor.cuda()

    DATASET = DFAUSTDataset(data_root=args.data_dir, split='test')
    DataLoader = torch.utils.data.DataLoader(DATASET, batch_size=1, shuffle=False, num_workers=0,
                                          pin_memory=True, drop_last=False,
                                          worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    
    with torch.no_grad():
        regressor = regressor.eval()

        for i, (points, knn, f_name) in tqdm(enumerate(DataLoader), total=len(DataLoader), smoothing=0.9):
            points = torch.Tensor(points.float()).cuda().contiguous()
            knn = knn.long().squeeze().cuda()
            f_name = f_name[0]

            v_num = points.shape[1]

            l = torch.zeros((v_num, args.k-1))
            minv = torch.zeros(v_num)

            # extract point cloud feature
            pt_features  = regressor(points, None, None, feature_only=True)

            points = points.repeat(args.n_points, 1, 1)
            pt_features = pt_features.repeat(args.n_points, 1, 1)

            for j in range(0, v_num, args.n_points):

                ref_pt = torch.arange(j, min(j+args.n_points, v_num)).long().cuda() 

                if ref_pt.shape[0] != args.n_points:
                    points = points[:ref_pt.shape[0]]
                    pt_features = pt_features[:ref_pt.shape[0]]

                l_pred, minv_pred, _, _= regressor(points, ref_pt, knn[j:min(j+args.n_points, v_num)], pt_features)

                for idx in range(l_pred.shape[0]):

                    current_ref_pt = ref_pt[idx]

                    l[current_ref_pt] = l_pred.squeeze()[idx].cpu()
                    minv[current_ref_pt] = minv_pred[idx].cpu()

                if j % 30 == 0:
                    print(l_pred[0])

        l = l.detach().numpy()
        minv = minv.detach().numpy()

        np.save(os.path.join(args.out_dir, f_name+'_l.npy'), l)
        np.save(os.path.join(args.out_dir, f_name+'_minv.npy'), minv)
     
    pool = Pool()
    pool.map(postprocess_l_minv, range(len(os.listdir(os.path.join(args.data_dir, 'test', 'V')))))
    
      
if __name__ == '__main__':
    args = parse_args()
    main()
