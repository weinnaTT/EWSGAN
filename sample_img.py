from __future__ import absolute_import, division, print_function

import cfg
import archs
import datasets
from network import train, my_validate, LinearLrDecay, load_params, copy_params, save_dis_images
from utils.utils import set_log_dir, save_checkpoint, save_is_checkpoint, create_logger, count_parameters_in_MB
# from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception
from utils.flop_benchmark import print_FLOPs
from archs.fully_super_network import Generator, Discriminator
import torch
import os
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
from archs.fully_super_network import simple_Discriminator
from torchvision.utils import make_grid, save_image

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)

    # set visible GPU ids
    if len(args.gpu_ids) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    # set TensorFlow environment for evaluation (calculate IS and FID)
    # _init_inception()
    inception_path = check_or_download_inception('./tmp/imagenet/')
    create_inception_graph(inception_path)

    # the first GPU in visible GPUs is dedicated for evaluation (running Inception model)
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for id in range(len(str_ids)):
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 1:
        args.gpu_ids = args.gpu_ids[1:]
    else:
        args.gpu_ids = args.gpu_ids
    genotype_G = np.load(os.path.join('exps', args.genotype_of_G))

    basemodel_gen = Generator(args, genotype_G)
    gen_net = torch.nn.DataParallel(basemodel_gen, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])

    if args.use_basemodel_D:
        basemodel_dis = simple_Discriminator()
    else:
        genotype_D = np.load(os.path.join('exps', args.genotypes_exp))
        basemodel_dis = Discriminator(args, genotype_D)

    dis_net = torch.nn.DataParallel(basemodel_dis, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])

    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    gen_net.apply(weights_init)
    dis_net.apply(weights_init)

    # set up data_loader
    dataset = datasets.ImageDataset(args)
    train_loader = dataset.train

    # epoch number for dis_net
    args.max_epoch_D = args.max_epoch_G * args.n_critic
    if args.max_iter_G:
        args.max_epoch_D = np.ceil(args.max_iter_G * args.n_critic / len(train_loader))
    max_iter_D = args.max_epoch_D * len(train_loader)

    # set optimizer
    gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                     args.g_lr, (args.beta1, args.beta2))
    dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                     args.d_lr, (args.beta1, args.beta2))
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, max_iter_D)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, max_iter_D)

    # fid stat
    if args.dataset.lower() == 'cifar10':
        fid_stat = './fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'stl10':
        fid_stat = './fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    else:
        raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')
    assert os.path.exists(fid_stat)

    # initial
    gen_avg_param = copy_params(gen_net)
    start_epoch = 0
    best_fid = 1e4
    best_is = 0.
    # set writer
    if args.checkpoint:
        # resuming
        print(f'=> resuming from {args.checkpoint}')
        assert os.path.exists(os.path.join('exps', args.checkpoint))
        checkpoint_file = os.path.join('exps', args.checkpoint, 'Model', 'checkpoint_best.pth')
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch']
        best_fid = checkpoint['best_fid']
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
        avg_gen_net = deepcopy(gen_net)
        avg_gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        gen_avg_param = copy_params(avg_gen_net)
        del avg_gen_net

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
    else:
        # create new log dir
        assert args.exp_name
        args.path_helper = set_log_dir('exps', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])

    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }
    # model size
    logger.info('Param size of G = %fMB', count_parameters_in_MB(gen_net))
    logger.info('Param size of D = %fMB', count_parameters_in_MB(dis_net))
    print_FLOPs(basemodel_gen, (1, args.latent_dim), logger)
    print_FLOPs(basemodel_dis, (1, 3, args.img_size, args.img_size), logger)
    checkpoint_file = os.path.join('exps', 'model_path', 'checkpoint_best_is_stl.pth')
    checkpoint = torch.load(checkpoint_file)
    # gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
    # gen_net.load_state_dict(checkpoint['gen_state_dict'])
    gen_net.load_state_dict(checkpoint['gen_state_dict'])
    avg_gen_net = deepcopy(gen_net)
    avg_gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
    gen_avg_param = copy_params(avg_gen_net)
    for i in range(100):
        # save_dis_images(args, gen_net, i)
        gen_net = gen_net.eval()
        fid_buffer_dir = os.path.join(
            args.path_helper['sample_path'], 'fid_buffer')
        os.makedirs(fid_buffer_dir, exist_ok=True)
        display_noise = torch.cuda.FloatTensor(np.random.normal(
            0, 1, (64, args.latent_dim)))
        display = gen_net(display_noise).cpu()
        grid_net = (make_grid(display) + 1) / 2
        save_image(grid_net, os.path.join(fid_buffer_dir, '%d.png' % i))


if __name__ == '__main__':
    main()
