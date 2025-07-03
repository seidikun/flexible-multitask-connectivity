import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(r'C:\Users\User\Documents\GitHub\flexible_multitask\stepnet')
from tools_lnd import load_X_from_model_dir

REPO_DIR          = r'C:\Users\User\Documents\GitHub\flexible-multitask-connectivity'
TRAINED_NETS      = os.path.join(REPO_DIR, 'trained_nets')
rules             = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti', 'delaydm1', 'delaydm2',
                'contextdelaydm1', 'contextdelaydm2', 'multidelaydm', 'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']
colormap          = plt.cm.nipy_spectral

def mse(y_pred, y_true):
    return np.nanmean((y_pred - y_true) ** 2)

n_rules           = len(rules)
n_per_row         = 3  # How many test rules per row (you can change)
n_rows            = int(np.ceil(n_rules / n_per_row))
n_cols            = n_per_row * 2  # 2 columns for each test_rule: pred and target

for i_net, train_rule in enumerate(rules):
    folder_name   = f"LeakyRNN_softplus_diag_128n_{train_rule}"
    model_dir     = os.path.join(TRAINED_NETS, folder_name, 'seed0')

    fig, axes     = plt.subplots(n_rows, n_cols, figsize=(4*n_per_row, 2.4*n_rows), constrained_layout=True)
    if n_rows == 1:
        axes      = np.expand_dims(axes, axis=0)
    axes          = axes.reshape(n_rows, n_cols)
    corr_img_path = None

    for j_rule, test_rule in enumerate(rules):
        h_tf, y_hat, hparams, trial = load_X_from_model_dir(model_dir, test_rule, noise=False)

        mse_score = mse(y_hat, trial.y)
        trials    = h_tf.shape[1]
        colors    = [colormap(x) for x in np.linspace(0, 1, trials)]

        # Find subplot position
        row       = j_rule // n_per_row
        col_pred  = (j_rule % n_per_row) * 2
        col_tgt   = col_pred + 1

        ax_pred   = axes[row, col_pred]
        ax_tgt    = axes[row, col_tgt]

        # Plot predictions and targets
        for i in range(trials):
            ax_pred.plot(y_hat[:, i, 1], y_hat[:, i, 2], color=colors[i], lw=0.7, alpha=0.8)
            ax_tgt.plot(trial.y[:, i, 1], trial.y[:, i, 2], color=colors[i], lw=0.7, alpha=0.8)
        ax_pred.set_xticks([]); ax_pred.set_yticks([])
        ax_tgt.set_xticks([]); ax_tgt.set_yticks([])
        ax_pred.set_title(f'{test_rule}\nNet', fontsize=8)
        ax_tgt.set_title('Target', fontsize=8)

        # Highlight where test_rule == train_rule
        if test_rule == train_rule:
            for ax in (ax_pred, ax_tgt):
                for spine in ax.spines.values():
                    spine.set_edgecolor('crimson')
                    spine.set_linewidth(2.5)
                ax.set_facecolor('#ffe0e0')
            ax_pred.set_title(f'{test_rule}\nNet [MSE: {mse_score:.2e}]', fontsize=10, weight='bold', color='crimson')
            ax_tgt.set_title('Target\n[train == test]', fontsize=10, weight='bold', color='crimson')

            # Save variables and correlation matrix
            save_dir = os.path.join(model_dir, f'eval_{test_rule}')
            os.makedirs(save_dir, exist_ok=True)
            np.savez(os.path.join(save_dir, 'vars.npz'), h_tf=h_tf, y_hat=y_hat, hparams=hparams, y=trial.y)
            corr_mat = np.corrcoef(h_tf.reshape(-1, h_tf.shape[2]), rowvar=False)
            pd.DataFrame(corr_mat).to_csv(os.path.join(save_dir, 'correlation_matrix.csv'), header=False, index=False)
           
            # Correlation image
            fig_corr, ax_corr = plt.subplots(figsize=(4, 3))
            im       = ax_corr.imshow(corr_mat, cmap='bwr', vmin=-1, vmax=1)
            ax_corr.set_title(f'Correlation Matrix\n{train_rule}', fontsize=10)
            ax_corr.set_xlabel('Neuron')
            ax_corr.set_ylabel('Neuron')
            plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
            plt.tight_layout()
            corr_img_path = os.path.join(save_dir, 'correlation_matrix.png')
            plt.savefig(corr_img_path, dpi=200)
            plt.close(fig_corr)

    fig.suptitle(f'Network trained on: {train_rule}', fontsize=14, y=1.01)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(os.path.join(model_dir, f'cross_task_grid_{train_rule}.png'), dpi=180)
    plt.close(fig)

    # Optionally, save correlation matrix figure separately
    if corr_img_path is not None:
        img = plt.imread(corr_img_path)
        plt.figure(figsize=(5, 4))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Correlation matrix - {train_rule}', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, f'corr_matrix_view_{train_rule}.png'), dpi=160)
        plt.close()