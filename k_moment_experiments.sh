#!/bin/bash


################################################################################
# GENERATIVE MODELS.

# Run cmd_k.
if [ "$1" == 1 ]; then
  python multivariate_privacy.py --model_type='cmd_gan' --data_file='' \
    --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
    --gen_num=100 --width=6 --depth=6 --z_dim=2 --learning_rate=1e-3 \
    --max_step=150000 \
    --k_moments=3 --tag='cmd_k'
fi

# Run cmd_k_minus_k_plus_1.
if [ "$1" == 2 ]; then
  python multivariate_privacy.py --model_type='cmd_gan' --data_file='' \
  --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
  --gen_num=100 --width=6 --depth=6 --z_dim=2 --learning_rate=1e-3 \
  --max_step=150000 \
  --k_moments=3 --cmd_variation='minus_k_plus_1' --tag='cmd_k_minus_k_plus_1'
fi

# Run cmd_k_minus_mmd.
if [ "$1" == 3 ]; then
  python multivariate_privacy.py --model_type='cmd_gan' --data_file='' \
  --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
  --gen_num=100 --width=6 --depth=6 --z_dim=2 --learning_rate=1e-3 \
  --max_step=150000 \
  --k_moments=3 --cmd_variation='minus_mmd' --tag='cmd_k_minus_mmd'
fi

# Run prog_cmd.
if [ "$1" == 4 ]; then
  python multivariate_privacy.py --model_type='cmd_gan' --data_file='' \
  --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
  --gen_num=100 --width=6 --depth=6 --z_dim=2 --learning_rate=1e-3 \
  --max_step=150000 \
  --k_moments=3 --cmd_variation='prog_cmd' --tag='prog_cmd'
fi

# Run mmd_gan_simple.
if [ "$1" == 5 ]; then
  python multivariate_privacy.py --model_type='mmd_gan_simple' --data_file='' \
  --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
  --gen_num=100 --width=6 --depth=6 --z_dim=2 --learning_rate=1e-3 \
  --max_step=150000 \
  --k_moments=3 --tag='mmd_gan_simple'
fi

# Run mmd_gan.
if [ "$1" == 6 ]; then
  python multivariate_privacy.py --model_type='mmd_gan' --data_file='' \
  --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
  --gen_num=100 --width=6 --depth=6 --z_dim=2 --learning_rate=1e-3 \
  --max_step=150000 \
  --k_moments=3 --tag='mmd_gan'
fi


###############################################################################
# AUTOENCODER MODELS.

# Run ae_partition_data.
if [ "$1" == 7 ]; then
  python multivariate_privacy.py --model_type='ae_base' --data_file='' \
  --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
  --gen_num=100 --width=6 --depth=6 --z_dim=2 --learning_rate=1e-3 \
  --max_step=150000 \
  --k_moments=3 --ae_variation='partition_ae_data' \
  --tag='ae_partition_ae_data'
fi

# Run ae_partition_encodings.
if [ "$1" == 8 ]; then
  python multivariate_privacy.py --model_type='ae_base' --data_file='' \
  --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
  --gen_num=100 --width=6 --depth=6 --z_dim=2 --learning_rate=1e-3 \
  --max_step=150000 \
  --k_moments=3 --ae_variation='partition_enc_enc' \
  --tag='ae_partition_enc_enc'
fi

# Run ae_subset.
if [ "$1" == 9 ]; then
  python multivariate_privacy.py --model_type='ae_base' --data_file='' \
  --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
  --gen_num=100 --width=6 --depth=6 --z_dim=2 --learning_rate=1e-3 \
  --max_step=150000 \
  --k_moments=3 --ae_variation='subset' --tag='ae_subset'
fi

# Run ae_mmd.
if [ "$1" == 10 ]; then
  python multivariate_privacy.py --model_type='ae_base' --data_file='' \
  --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
  --gen_num=100 --width=6 --depth=6 --z_dim=2 --learning_rate=1e-3 \
  --max_step=150000 \
  --k_moments=3 --ae_variation='mmd' --tag='ae_mmd'
fi

# Run ae_cmd_k.
if [ "$1" == 11 ]; then
  python multivariate_privacy.py --model_type='ae_base' --data_file='' \
  --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
  --gen_num=100 --width=6 --depth=6 --z_dim=2 --learning_rate=1e-3 \
  --max_step=150000 \
  --k_moments=3 --ae_variation='cmd_k' --tag='ae_cmd_k'
fi

# Run ae_cmd_k_minus_k_plus_1.
if [ "$1" == 12 ]; then
  python multivariate_privacy.py --model_type='ae_base' --data_file='' \
  --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
  --gen_num=100 --width=6 --depth=6 --z_dim=2 --learning_rate=1e-3 \
  --max_step=150000 \
  --k_moments=3 --ae_variation='cmd_k_minus_k_plus_1' \
  --tag='ae_cmd_k_minus_k_plus_1'
fi

###############################################################################
# Run Tensorboard for the above summary results.
if [ "$1" == "tb" ]; then
  rm -rf logs/logs_all; mkdir logs/logs_all; \
    # Generative base.
    cp -r logs/logs_mmd_gan_simple/summary logs/logs_all/mmd_gan_simple; \
    cp -r logs/logs_mmd_gan/summary logs/logs_all/mmd_gan; \
    cp -r logs/logs_cmd_k/summary logs/logs_all/cmd_k; \
    cp -r logs/logs_cmd_k_minus_k_plus_1/summary logs/logs_all/cmd_k_minus_k_plus_1; \
    cp -r logs/logs_cmd_k_minus_mmd/summary logs/logs_all/cmd_k_minus_mmd; \
    cp -r logs/logs_prog_cmd/summary logs/logs_all/prog_cmd; \
    # AE base.
    cp -r logs/logs_ae_partition_ae_data/summary logs/logs_all/ae_partition_ae_data; \
    cp -r logs/logs_ae_partition_enc_enc/summary logs/logs_all/ae_partition_enc_enc; \
    cp -r logs/logs_ae_subset/summary logs/logs_all/ae_subset; \
    cp -r logs/logs_ae_mmd/summary logs/logs_all/ae_mmd; \
    cp -r logs/logs_ae_cmd_k/summary logs/logs_all/ae_cmd_k; \
    cp -r logs/logs_ae_cmd_k_minus_k_plus_1/summary logs/logs_all/ae_cmd_k_minus_k_plus_1; \
    tensorboard --logdir=logs/logs_all --port 6006
fi
