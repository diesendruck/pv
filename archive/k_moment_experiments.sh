#!/bin/bash


################################################################################
# GENERATIVE MODELS.

# Run cmd_k.
if [ "$1" == 1 ]; then
  python multivariate_privacy.py --model_type='cmd_gan' --data_file='' \
    --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
    --gen_num=100 --width=6 --depth=6 --z_dim=2 --learning_rate=1e-4 \
    --max_step=150000 \
    --k_moments=5 --tag='cmd_k' \
    &> logs/output_cmd_k.txt
fi

# Run cmd_k_minus_k_plus_1.
if [ "$1" == 2 ]; then
  python multivariate_privacy.py --model_type='cmd_gan' --data_file='' \
    --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
    --gen_num=100 --width=6 --depth=6 --z_dim=2 --learning_rate=1e-4 \
    --max_step=150000 \
    --k_moments=5 --cmd_variation='minus_k_plus_1' --tag='cmd_k_minus_k_plus_1' \
    &> logs/output_cmd_k_minus_k_plus_1.txt
fi

# Run cmd_k_minus_mmd.
if [ "$1" == 3 ]; then
  python multivariate_privacy.py --model_type='cmd_gan' --data_file='' \
  --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
  --gen_num=100 --width=6 --depth=6 --z_dim=2 --learning_rate=1e-4 \
  --max_step=150000 \
  --k_moments=5 --cmd_variation='minus_mmd' --tag='cmd_k_minus_mmd' \
    &> logs/output_cmd_k_minus_mmd.txt
fi

# Run prog_cmd.
if [ "$1" == 4 ]; then
  python multivariate_privacy.py --model_type='cmd_gan' --data_file='' \
    --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
    --gen_num=100 --width=6 --depth=6 --z_dim=2 --learning_rate=1e-4 \
    --max_step=150000 \
    --k_moments=5 --cmd_variation='prog_cmd' --tag='prog_cmd' \
    &> logs/output_prog_cmd.txt
fi

# Run mmd_gan_simple.
if [ "$1" == 5 ]; then
  python multivariate_privacy.py --model_type='mmd_gan_simple' --data_file='' \
    --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
    --gen_num=100 --width=6 --depth=6 --z_dim=2 --learning_rate=1e-4 \
    --max_step=150000 \
    --k_moments=5 --tag='mmd_gan_simple' \
    &> logs/output_mmd_gan_simple.txt
fi

# Run mmd_gan.
if [ "$1" == 6 ]; then
  python multivariate_privacy.py --model_type='mmd_gan' --data_file='' \
    --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
    --gen_num=100 --width=6 --depth=6 --z_dim=2 --learning_rate=1e-4 \
    --max_step=150000 \
    --k_moments=5 --tag='mmd_gan' \
    &> logs/output_mmd_gan.txt
fi


###############################################################################
# AUTOENCODER MODELS.

# Run ae_partition_data.
if [ "$1" == 7 ]; then
  python multivariate_privacy.py --model_type='ae_base' --data_file='' \
    --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
    --gen_num=100 --width=6 --depth=6 --z_dim=2 --learning_rate=1e-4 \
    --max_step=150000 \
    --k_moments=5 --ae_variation='partition_ae_data' \
    --tag='ae_partition_ae_data' \
    &> logs/output_ae_partition_ae_data.txt
fi

# Run ae_partition_encodings.
if [ "$1" == 8 ]; then
  python multivariate_privacy.py --model_type='ae_base' --data_file='' \
    --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
    --gen_num=100 --width=6 --depth=6 --z_dim=2 --learning_rate=1e-4 \
    --max_step=150000 \
    --k_moments=5 --ae_variation='partition_enc_enc' \
    --tag='ae_partition_enc_enc' \
    &> logs/output_ae_partition_enc_enc.txt
fi

# Run ae_subset.
if [ "$1" == 9 ]; then
  python multivariate_privacy.py --model_type='ae_base' --data_file='' \
    --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
    --gen_num=100 --width=6 --depth=6 --z_dim=2 --learning_rate=1e-4 \
    --max_step=150000 \
    --k_moments=5 --ae_variation='subset' --tag='ae_subset' \
    &> logs/output_ae_subset.txt
fi

# Run ae_mmd.
if [ "$1" == 10 ]; then
  python multivariate_privacy.py --model_type='ae_base' --data_file='' \
    --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
    --gen_num=100 --width=6 --depth=6 --z_dim=2 --learning_rate=1e-4 \
    --max_step=150000 \
    --k_moments=5 --ae_variation='mmd' --tag='ae_mmd' \
    &> logs/output_ae_mmd.txt
fi

# Run ae_cmd_k.
if [ "$1" == 11 ]; then
  python multivariate_privacy.py --model_type='ae_base' --data_file='' \
    --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
    --gen_num=100 --width=6 --depth=6 --z_dim=2 --learning_rate=1e-4 \
    --max_step=150000 \
    --k_moments=5 --ae_variation='cmd_k' --tag='ae_cmd_k' \
    &> logs/output_ae_cmd_k.txt
fi

# Run ae_cmd_k_minus_k_plus_1.
if [ "$1" == 12 ]; then
  python multivariate_privacy.py --model_type='ae_base' --data_file='' \
    --log_step=1000 --data_dimension=1 --data_num=5000 --batch_size=100 \
    --gen_num=100 --width=6 --depth=6 --z_dim=2 --learning_rate=1e-4 \
    --max_step=150000 \
    --k_moments=5 --ae_variation='cmd_k_minus_k_plus_1' \
    --tag='ae_cmd_k_minus_k_plus_1' \
    &> logs/output_ae_cmd_k_minus_k_plus_1.txt
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
    #cp -r logs/logs_k3_switch/summary logs/logs_all/k3_switch; \
    #cp -r logs/logs_k5_switch/summary logs/logs_all/k5_switch; \
    #cp -r logs/logs_k3_base/summary logs/logs_all/k3_base; \
    #cp -r logs/logs_k5_base/summary logs/logs_all/k5_base; \
    #cp -r logs/logs_cmd5/summary logs/logs_all/cmd5; \
    cp -r logs/logs_cmd3/summary logs/logs_all/cmd3; \
    cp -r logs/logs_cmd3_v2/summary logs/logs_all/cmd3_v2; \
    cp -r logs/logs_mmd_cmd3_1/summary logs/logs_all/mmd_cmd3_1; \
    cp -r logs/logs_mmd_cmd3_2/summary logs/logs_all/mmd_cmd3_2; \
    cp -r logs/logs_mmd_cmd3_3/summary logs/logs_all/mmd_cmd3_3; \
    cp -r logs/logs_mmd_cmd3_4/summary logs/logs_all/mmd_cmd3_4; \
    #
    cp -r logs/logs_cmd5/summary logs/logs_all/cmd5; \
    cp -r logs/logs_cmd5_v2/summary logs/logs_all/cmd5_v2; \
    cp -r logs/logs_mmd_cmd5_1/summary logs/logs_all/mmd_cmd5_1; \
    cp -r logs/logs_mmd_cmd5_2/summary logs/logs_all/mmd_cmd5_2; \
    cp -r logs/logs_mmd_cmd5_3/summary logs/logs_all/mmd_cmd5_3; \
    cp -r logs/logs_mmd_cmd5_4/summary logs/logs_all/mmd_cmd5_4; \
    #
    cp -r logs/logs_cmd7/summary logs/logs_all/cmd7; \
    cp -r logs/logs_cmd7_v2/summary logs/logs_all/cmd7_v2; \
    cp -r logs/logs_mmd_cmd7_1/summary logs/logs_all/mmd_cmd7_1; \
    cp -r logs/logs_mmd_cmd7_2/summary logs/logs_all/mmd_cmd7_2; \
    cp -r logs/logs_mmd_cmd7_3/summary logs/logs_all/mmd_cmd7_3; \
    cp -r logs/logs_mmd_cmd7_4/summary logs/logs_all/mmd_cmd7_4; \
    # AE base.
    cp -r logs/logs_ae_partition_ae_data/summary logs/logs_all/ae_partition_ae_data; \
    cp -r logs/logs_ae_partition_enc_enc/summary logs/logs_all/ae_partition_enc_enc; \
    cp -r logs/logs_ae_subset/summary logs/logs_all/ae_subset; \
    cp -r logs/logs_ae_mmd/summary logs/logs_all/ae_mmd; \
    cp -r logs/logs_ae_cmd_k/summary logs/logs_all/ae_cmd_k; \
    cp -r logs/logs_ae_cmd_k_minus_k_plus_1/summary logs/logs_all/ae_cmd_k_minus_k_plus_1; \
    tensorboard --logdir=logs/logs_all --port 6006
fi
