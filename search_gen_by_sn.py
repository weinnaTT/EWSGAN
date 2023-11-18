from __future__ import absolute_import, division, print_function
import search_cfg
import archs
import datasets
from trainer.trainer_generator import GenTrainer
from trainer.trainer_utils import LinearLrDecay
from utils.utils import set_log_dir, save_checkpoint, create_logger, count_parameters_in_MB
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception
from utils.flop_benchmark import print_FLOPs
from archs.super_network import Generator, Discriminator
from archs.fully_super_network import simple_Discriminator
from algorithms.search_algs import GanAlgorithm
import torch
import os
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
# import copy
from copy import deepcopy
from pytorch_gan_metrics import get_inception_score_and_fid

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False


def main():
    args = search_cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)
    # set visible GPU ids
    if len(args.gpu_ids) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
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

    # genotype G
    gan_alg = GanAlgorithm(args)

    # import network from genotype
    basemodel_gen = Generator(args)  # 这里为超网
    gen_net = torch.nn.DataParallel(
        basemodel_gen, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])
    basemodel_dis = simple_Discriminator()
    dis_net = torch.nn.DataParallel(
        basemodel_dis, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])

    # weight init

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError(
                    '{} unknown inital type'.format(args.init_type))
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
    if args.max_iter_G:  # 向上取整
        args.max_epoch_D = np.ceil(
            args.max_iter_G * args.n_critic / len(train_loader))
    max_iter_D = args.max_epoch_D * len(train_loader)
    # set TensorFlow environment for evaluation (calculate IS and FID)
    # _init_inception()

    inception_path = check_or_download_inception('./tmp/imagenet/')
    create_inception_graph(inception_path)

    if args.dataset.lower() == 'cifar10':
        fid_stat = './fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'stl10':
        fid_stat = './fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    else:
        raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')
    assert os.path.exists(fid_stat)
    gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                     args.g_lr, (args.beta1, args.beta2))
    dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                     args.d_lr, (args.beta1, args.beta2))
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, max_iter_D)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, max_iter_D)

    # initial
    start_epoch = 0
    best_fid = 1e4
    # set writer

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
    # genotype_fixG = gan_alg.search(remove=False)
    genotype_fixG = np.load(os.path.join('exps', 'best_G.npy'))
    # genotype_fixG = gan_alg.sample_zero()

    # read supernetG
    ckpt = torch.load(os.path.join('exps', args.model_name))
    gen_net.load_state_dict(ckpt['weight_G'])
    # gan_alg.Normal_G_fixed = deepcopy(ckpt['normal_G_fixed'])
    # gan_alg.Up_G_fixed = deepcopy(ckpt['up_G_fixed'])
    up_temp = (np.load(os.path.join('exps', 'Up_G_fixed.npy'))).tolist()
    normal_temp = (np.load(os.path.join('exps', 'Normal_G_fixed.npy'))).tolist()
    gan_alg.Normal_G_fixed = deepcopy(normal_temp)
    gan_alg.Up_G_fixed = deepcopy(up_temp)
    trainer_gen = GenTrainer(args, gen_net, dis_net, gen_optimizer,
                             dis_optimizer, train_loader, gan_alg, None,
                             genotype_fixG)
    best_genotypes = None
    # search genarator
    ll = []
    for i in range(args.num_individual):
        ll.append([])
        while (True):
            a1 = gan_alg.search()
            if trainer_gen.judege_model_size(a1, limit=args.max_model_size):
                break
        ll[i] = a1
    population = np.stack(ll)

    record_is = []
    record_fid = []

    for ii in tqdm(range(args.Total_evolutionary_algebra), desc='search genearator using evo alg'):
        population, pop_selected, a, b, is_record, fid_record = trainer_gen.my_search_evolv2(population, fid_stat, ii)
        record_is.append(is_record.tolist())
        record_fid.append(fid_record.tolist())

    for index, geno in enumerate(pop_selected):
        file_path = os.path.join(args.path_helper['ckpt_path'],
                                 "best_gen_{}.npy".format(str(index)))
        np.save(file_path, geno)
    file = open("IS_record.txt", 'w')
    for fp in record_is:
        file.write(str(fp))
        file.write('\n')
    file.close()

    file = open("FID_record.txt", 'w')
    for fp in record_fid:
        file.write(str(fp))
        file.write('\n')
    file.close()


if __name__ == '__main__':
    main()
