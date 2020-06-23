#!/bin/bash

python momabs/make_plots.py runs/bandit_type_gaussian/objectives_2/utility_type_random_polynomial/horizon_1500/repeat_1/sig_threshold_0.01/utility_std_0.1 runs/bandit_type_gaussian/objectives_2/utility_type_random_polynomial/horizon_1500/repeat_1/sig_threshold_0.01/utility_std_0.5 runs/bandit_type_gaussian/objectives_2/utility_type_random_polynomial/horizon_1500/repeat_1/sig_threshold_0.01/utility_std_1.0 --keys "e=0.1" "e=0.5" "e=1." --comp noise

python momabs/make_plots.py runs/bandit_type_gaussian/objectives_2/utility_type_random_polynomial/horizon_1500/repeat_1/sig_threshold_0.01/utility_std_0.1 runs/bandit_type_gaussian/objectives_2/utility_type_random_polynomial/horizon_1500/repeat_1/sig_threshold_1.0/utility_std_0.1 --keys "sig" "no sig" --comp sig

python momabs/make_plots.py runs/bandit_type_gaussian/objectives_4/utility_type_random_polynomial/horizon_1500/repeat_1/sig_threshold_0.01/utility_std_0.1 runs/bandit_type_gaussian/objectives_4/utility_type_random_polynomial/horizon_1500/repeat_1/sig_threshold_1.0/utility_std_0.1 --keys "sig" "no sig" --comp sig

python momabs/make_plots.py runs/bandit_type_gaussian/objectives_6/utility_type_random_polynomial/horizon_1500/repeat_1/sig_threshold_0.01/utility_std_0.1 runs/bandit_type_gaussian/objectives_6/utility_type_random_polynomial/horizon_1500/repeat_1/sig_threshold_1.0/utility_std_0.1 --keys "sig" "no sig" --comp sig

python momabs/make_plots.py runs/bandit_type_gaussian/objectives_2/utility_type_random_polynomial/horizon_1500/repeat_1/sig_threshold_0.01/utility_std_0.1/ --keys "" --comp cool
