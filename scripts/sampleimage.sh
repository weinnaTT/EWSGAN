
python sample_img.py \
--gpu_ids 0 \
--num_workers 8 \
--gen_bs 128 \
--dis_bs 64 \
--dataset Cifar10 \
--bottom_width 6 \
--img_size 48 \
--max_epoch_G 120 \
--n_critic 5 \
--arch arch_cifar10 \
--draw_arch False \
--genotypes_exp cifar10_D.npy \
--latent_dim 120 \
--gf_dim 256 \
--df_dim 128 \
--g_lr 0.0002 \
--d_lr 0.0002 \
--beta1 0.0 \
--beta2 0.9 \
--init_type xavier_uniform \
--val_freq 5 \
--num_eval_imgs 50000 \
--exp_name arch_train_stl10 \
--data_path ~/datasets/cifar10 \
--cr 0 \
--genotype_of_G my_best_gen.npy \
--use_basemodel_D False