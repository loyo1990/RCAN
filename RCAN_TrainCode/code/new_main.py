import argparse

import torch

import model
import scipy.misc as ssc

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--scale', default='4',type=int,
                    help='super resolution scale')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--model', default='RCAN',
                    help='model name')
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=20,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='checkpoint path')


if __name__ == '__main__':
    args = parser.parse_args()
    net = model.Model(args)
    if args.checkpoint:
        net.load(args.checkpoint, args.cpu)

    x = ssc.imread('./frame_0001.png')
    x = ssc.imresize(x, (48,48))
    x = x.transpose((2, 0, 1))[None, :,:,:]
    x = x
    x = torch.from_numpy(x).float()
    x = net(x, 0)
    x = x.data[0].numpy().transpose(1,2,0)
    ssc.imsave('./result.png', x)
