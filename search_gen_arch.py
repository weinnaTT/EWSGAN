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
    basemodel_gen = Generator(args)
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
    max_epoch_for_D = args.my_max_epoch_G * args.n_critic
    args.max_epoch_D = args.my_max_epoch_G * args.n_critic
    max_iter_D = max_epoch_for_D * len(train_loader)
    # set TensorFlow environment for evaluation (calculate IS and FID)
    # _init_inception()

    # inception_path = check_or_download_inception('./tmp/imagenet/')
    # create_inception_graph(inception_path)
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
    trainer_gen = GenTrainer(args, gen_net, dis_net, gen_optimizer,
                             dis_optimizer, train_loader, gan_alg, None,
                             genotype_fixG)
    best_genotypes = None
    is_mean_best = 0.0
    fid_mean_best = 999.0

    temp_model_size = args.max_model_size
    args.max_model_size = 999

    epoch_record = []
    is_record = []
    fid_record = []

    # search genarator
    for epoch in tqdm(range(int(start_epoch), int(200)), desc='training the supernet_G:'):
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        trainer_gen.train(epoch, writer_dict, fid_stat, lr_schedulers)
        if epoch == 200:
            break
        if epoch >= 9999 and epoch % 5 == 0:
            now_is_max = 0
            now_fid_min = 999
            trainer_gen.clear_bag()
            population = np.stack([gan_alg.search() for i in range(12)], axis=0)

            for kk in tqdm(range(4),
                           desc='Evaluating of subnet performance using evolutionary algorithms'):
                population, pop_selected, is_mean, fid_mean, is_max, fid_min = trainer_gen.my_search_evol(population,
                                                                                                          fid_stat, kk)
                if is_max > now_is_max:
                    now_is_max = is_max
                if fid_min < now_fid_min:
                    now_fid_min = fid_min
            epoch_record.append(epoch)
            is_record.append(now_is_max)
            fid_record.append(now_fid_min)
            np.save('epoch_record_518.npy', np.array(epoch_record))
            np.save('is_record_518.npy', np.array(is_record))
            np.save('fid_record_518.npy', np.array(fid_record))

            trainer_gen.clear_bag()

            if is_mean > is_mean_best:
                is_mean_best = is_mean
                checkpoint_file = os.path.join(args.path_helper['ckpt_path'], 'gen_checkpoint_best_is.pt')
                ckpt = {'epoch': epoch,
                        'weight_G': gen_net.state_dict(),
                        'weight_D': dis_net.state_dict(),
                        'up_G_fixed': gan_alg.Up_G_fixed,
                        'normal_G_fixed': gan_alg.Normal_G_fixed,
                        }
                torch.save(ckpt, checkpoint_file)
                del ckpt
            if fid_mean < fid_mean_best:
                fid_mean_best = fid_mean
                checkpoint_file = os.path.join(args.path_helper['ckpt_path'], 'gen_checkpoint_best_fid.pt')
                ckpt = {'epoch': epoch,
                        'weight_G': gen_net.state_dict(),
                        'weight_D': dis_net.state_dict(),
                        'up_G_fixed': gan_alg.Up_G_fixed,
                        'normal_G_fixed': gan_alg.Normal_G_fixed,
                        }
                torch.save(ckpt, checkpoint_file)
                del ckpt

        if epoch == args.warmup * args.n_critic:  # Start discarding
            checkpoint_file = os.path.join(args.path_helper['ckpt_path'], 'gen_checkpoint_before_prune.pt')
            ckpt = {'epoch': epoch,
                    'weight_G': gen_net.state_dict(),
                    'weight_D': dis_net.state_dict(),
                    'up_G_fixed': gan_alg.Up_G_fixed,
                    'normal_G_fixed': gan_alg.Normal_G_fixed,
                    }
            torch.save(ckpt, checkpoint_file)
            del ckpt
            # trainer_gen.greedy_modify_fixed(42, fid_stat)
            trainer_gen.directly_modify_fixed(fid_stat)
            logger.info(
                f'gan_alg.Normal_G_fixed: {gan_alg.Normal_G_fixed}, gan_alg.Up_G_fixed: {gan_alg.Up_G_fixed},@ epoch {epoch}.')

    checkpoint_file = os.path.join(args.path_helper['ckpt_path'], 'supernet_gen.pt')
    ckpt = {
        'weight_G': gen_net.state_dict(),
        'up_G_fixed': gan_alg.Up_G_fixed,
        'normal_G_fixed': gan_alg.Normal_G_fixed,
    }
    torch.save(ckpt, checkpoint_file)
    args.max_model_size = temp_model_size
    population = np.stack([gan_alg.search() for i in range(args.num_individual)], axis=0)
    for ii in tqdm(range(args.Total_evolutionary_algebra), desc='search genearator using evo alg'):
        population, pop_selected, is_mean, fid_mean, _, _ = trainer_gen.my_search_evolv2(population, fid_stat, ii)
    for index, geno in enumerate(pop_selected):
        file_path = os.path.join(args.path_helper['ckpt_path'],
                                 "best_gen_{}.npy".format(str(index)))
        np.save(file_path, geno)


if __name__ == '__main__':
    main()
