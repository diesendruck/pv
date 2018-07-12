#!/bin/bash

# Run cmd_3.
if [ "$1" == 1 ]; then
  python multivariate_privacy.py --model_type='cmd_gan' --data_file='' \
    --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
    --gen_num=100 --width=3 --depth=3 --z_dim=2 --learning_rate=1e-3 \
    --max_step=500000 \
    --k_moments=3 --tag='cmd_3'
fi

# Run cmd_3_minus_4.
if [ "$1" == 2 ]; then
  python multivariate_privacy.py --model_type='cmd_gan' --data_file='' \
  --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
  --gen_num=100 --width=3 --depth=3 --z_dim=2 --learning_rate=1e-3 \
  --max_step=500000 \
  --k_moments=3 --cmd_variation='minus_k_plus_1' --tag='cmd_3not4'
fi

# Run cmd_3_minus_mmd.
if [ "$1" == 3 ]; then
  python multivariate_privacy.py --model_type='cmd_gan' --data_file='' \
  --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
  --gen_num=100 --width=3 --depth=3 --z_dim=2 --learning_rate=1e-3 \
  --max_step=500000 \
  --k_moments=3 --cmd_variation='minus_mmd' --tag='cmd_3notMMD'
fi

# Run mmd_gan_simple.
if [ "$1" == 4 ]; then
  python multivariate_privacy.py --model_type='mmd_gan_simple' --data_file='' \
  --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
  --gen_num=100 --width=3 --depth=3 --z_dim=2 --learning_rate=1e-3 \
  --max_step=500000 \
  --k_moments=3 --tag='mmd_gan_simple'
fi

# Run ae_subset.
if [ "$1" == 5 ]; then
  python multivariate_privacy.py --model_type='ae_base' --data_file='' \
  --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
  --gen_num=100 --width=3 --depth=3 --z_dim=2 --learning_rate=1e-3 \
  --max_step=500000 \
  --k_moments=3 --ae_variation='subset' --tag='ae_subset'
fi

# Run ae_mmd.
if [ "$1" == 6 ]; then
  python multivariate_privacy.py --model_type='ae_base' --data_file='' \
  --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
  --gen_num=100 --width=3 --depth=3 --z_dim=2 --learning_rate=1e-3 \
  --max_step=500000 \
  --k_moments=3 --ae_variation='mmd' --tag='ae_mmd'
fi
# Run ae_cmd_k.
if [ "$1" == 7 ]; then
  python multivariate_privacy.py --model_type='ae_base' --data_file='' \
  --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
  --gen_num=100 --width=3 --depth=3 --z_dim=2 --learning_rate=1e-3 \
  --max_step=500000 \
  --k_moments=3 --ae_variation='cmd_k' --tag='ae_cmd_k'
fi

# Run ae_cmd_k_minus_k_plus_1.
if [ "$1" == 8 ]; then
  python multivariate_privacy.py --model_type='ae_base' --data_file='' \
  --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
  --gen_num=100 --width=3 --depth=3 --z_dim=2 --learning_rate=1e-3 \
  --max_step=500000 \
  --k_moments=3 --ae_variation='cmd_k_minus_k_plus_1' \
  --tag='ae_cmd_k_minus_k_plus_1'
fi

###############################################################################
# Run Tensorboard for the above summary results.
if [ "$1" == "tb" ]; then
  rm -rf logs_all; mkdir logs_all; \
    # Generative base.
    cp -r logs_cmd_3/summary logs_all/cmd_3; \
    cp -r logs_cmd_3not4/summary logs_all/3not4; \
    cp -r logs_cmd_3notMMD/summary logs_all/3notMMD; \
    cp -r logs_mmd_gan_simple/summary logs_all/mmd_gan_simple; \
    # AE base.
    cp -r logs_ae_subset/summary logs_all/ae_subset; \
    cp -r logs_ae_mmd/summary logs_all/ae_mmd; \
    cp -r logs_ae_cmd_k/summary logs_all/ae_cmd_k; \
    cp -r logs_ae_cmd_k_minus_k_plus_1/summary logs_all/ae_cmd_k_minus_k_plus_1; \
    tensorboard --logdir=logs_all --port 6006
fi
