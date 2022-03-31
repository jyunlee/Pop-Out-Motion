import os
import argparse
import numpy as np


''' dependency paths '''
manifold_build_dir = '../../dependencies/Manifold/build' 
ftetwild_build_dir = '../../dependencies/fTetWild/build' 


def parse_args():
    parser = argparse.ArgumentParser('Model') 
    parser.add_argument('--in_dir', type=str, default='../in_data', help='path to input data directory')
    parser.add_argument('--out_dir', type=str, default='../data', help='path to output data directory')

    return parser.parse_args() 


if __name__ == '__main__':

    args = parse_args()

    in_dir, out_dir, k = args.in_dir, args.out_dir, args.k

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for sub_dir in ['Mesh', 'V', 'L', 'Minv']:
        if not os.path.exists(os.path.join(out_dir, sub_dir)):
            os.mkdir(os.path.join(out_dir, sub_dir))


    for file in os.listdir(in_dir):

        if not file.endswith('.obj'):
            continue

        in_file = os.path.join(in_dir, file) 
        out_file = os.path.join(out_dir, 'Mesh', file)
        out_volume_file = os.path.join(out_dir, 'Mesh', file[:-4] + '.mesh')
    
        # run manifold
        os.system('%s %s %s' % (os.path.join(manifold_build_dir, 'manifold'), in_file, out_file))

        # normalize
        os.system('%s %s %s' % ('./build/normalize_bin', out_file, out_file))

        # run fTetWild
        os.system('%s --input %s --output %s' % (os.path.join(ftetwild_build_dir, 'FloatTetwild_bin'), out_file, out_volume_file))

        # calculate L and Minv
        os.system('%s %s %s %s' % ('./build/calc_l_minv_bin', out_volume_file, out_dir, file[:-4])) 

