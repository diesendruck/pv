
To run:
1. Make sure the two python files and one bash script are in one directory.
2. Remove line 13 in multivariate_privacy.py. (That line is unnecessary if step (1) is done.)
3. Choose a model from the pre-selected ones available in the k_moment_experiments.sh script. To run experiment 6, for example, type *bash k_moment_experiments.sh 6*.
4. See the results in the *logs/[TAG]* directory, and run tensorboard with *tensorboard --logdir=logs/logs_[TAG] --port 6006*.
