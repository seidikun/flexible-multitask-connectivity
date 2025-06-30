import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Add correct stepnet path
sys.path.append('./stepnet')
sys.path.append(r'C:\Users\User\Documents\GitHub\flexible_multitask\stepnet')

import task
from task import generate_trials, rules_dict
from network import Model, get_perf
import tools
import train

# ====== BASE REPOSITORY DIRECTORY ======
REPO_DIR = r'C:\Users\User\Documents\GitHub\flexible-multitask-connectivity'

# ====== COMMON PARAMETERS FOR ALL TASKS ======
common_params = {
    'rnn_type':      'LeakyRNN',
    'activation':    'softplus',
    'init':          'diag',
    'n_rnn':         128,
    'l2w':           -6,
    'l2h':           -6,
    'l1w':           0,
    'l1h':           0,
    'seed':          0,
    'lr':            -6,
    'sigma_rec':     1/20,
    'sigma_x':       2/20,
    'w_rec_coeff':   1,
}

# ====== LIST OF TASKS TO TRAIN (from rules_dict['all']) ======
tasks_to_train = rules_dict['all']

# ====== LOOP: TRAIN ONE NETWORK PER TASK ======
for task_name in tasks_to_train:

    print(f"\n=== Training network on task: {task_name} ===")

    # Prepare parameters for this run
    params = common_params.copy()
    params['ruleset'] = 'all'  # Use a generic name to avoid KeyError

    # Directory: .../experiments/LeakyRNN_softplus_diag_128n_fdgo/seed0/
    folder_name = f"{params['rnn_type']}_{params['activation']}_{params['init']}_{params['n_rnn']}n_{task_name}"
    exp_dir = os.path.join(REPO_DIR, 'experiments', folder_name, f"seed{params['seed']}")
    os.makedirs(exp_dir, exist_ok=True)

    # Save parameters
    with open(os.path.join(exp_dir, "params.txt"), "w") as f:
        for k, v in params.items():
            f.write(f"{k}: {v}\n")
    with open(os.path.join(exp_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=4)

    print(f"Parameters saved at: {os.path.join(exp_dir, 'params.txt')}")

    # ---- Train the model (single task per network) ----
    train.train(
        exp_dir,
        seed=params['seed'],
        max_steps=1e3,
        ruleset=params['ruleset'],   # <-- Not the task name! Use generic string
        rule_trains=[task_name],     # <-- Actual task being trained
        hp={
            'activation':         params['activation'],
            'w_rec_init':         params['init'],
            'n_rnn':              params['n_rnn'],
            'l1_h':               10**params['l1h'],
            'l2_h':               10**params['l2h'],
            'l1_weight':          10**params['l1w'],
            'l2_weight':          10**params['l2w'],
            'l2_weight_init':     0,
            'sigma_rec':          params['sigma_rec'],
            'sigma_x':            params['sigma_x'],
            'rnn_type':           params['rnn_type'],
            'use_separate_input': False,
            'learning_rate':      10**(params['lr']/2),
            'w_rec_coeff':        params['w_rec_coeff']
        },
        display_step=10000,
        rich_output=False
    )

    print(f"Experiment for task '{task_name}' saved at: {exp_dir}")
