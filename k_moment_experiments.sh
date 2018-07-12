#!/usr/bin/env bash

# cmd_3 
python multivariate_privacy.py --model_type='cmd_gan' --data_file='' \
  --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
  --gen_num=100 --width=3 --depth=3 --z_dim=2 --learning_rate=1e-3 \
  --max_step=500000 \
  --k_moments=3 --tag='cmd_3' &

# cmd_3_minus_4
python multivariate_privacy.py --model_type='cmd_gan' --data_file='' \
  --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
  --gen_num=100 --width=3 --depth=3 --z_dim=2 --learning_rate=1e-3 \
  --max_step=500000 \
  --k_moments=3 --cmd_variation='minus_k_plus_1' --tag='cmd_3not4' &

# cmd_3_minus_mmd
python multivariate_privacy.py --model_type='cmd_gan' --data_file='' \
  --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
  --gen_num=100 --width=3 --depth=3 --z_dim=2 --learning_rate=1e-3 \
  --max_step=500000 \
  --k_moments=3 --cmd_variation='minus_mmd' --tag='cmd_3notMMD' &

# mmd_gan_simple 
python multivariate_privacy.py --model_type='mmd_gan_simple' --data_file='' \
  --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
  --gen_num=100 --width=3 --depth=3 --z_dim=2 --learning_rate=1e-3 \
  --max_step=500000 \
  --k_moments=3 --tag='mmd_gan_simple' &

