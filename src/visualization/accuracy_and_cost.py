import os

import matplotlib.pyplot as plt


def plot_accuracy_and_cost(cost_list, accuracy_list, show=True, save_path=None):
    """
    Plot training cost and accuracy over epochs.

    Args:
        cost_list: List of cost values per epoch.
        accuracy_list: List of accuracy values per epoch (0-1 scale).
        show: Whether to call plt.show().
        save_path: Optional path to save the figure.

    Returns:
        fig: The matplotlib figure.
    """
    if len(cost_list) != len(accuracy_list):
        raise ValueError(
            f"cost_list and accuracy_list must have the same length "
            f"({len(cost_list)} vs {len(accuracy_list)})."
        )

    epochs = range(1, len(cost_list) + 1)
    accuracy_percent = [acc * 100 for acc in accuracy_list]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    cost_line, = ax1.plot(epochs, cost_list, color='tab:red', marker='o', label='Cost')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cost', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    accuracy_line, = ax2.plot(
        epochs, accuracy_percent, color='tab:blue', marker='o', label='Accuracy'
    )
    ax2.set_ylabel('Accuracy (%)', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_ylim(0, 105)

    fig.suptitle('Training Progress', fontsize=14, y=0.98)
    fig.legend(
        [cost_line, accuracy_line],
        ['Cost', 'Accuracy'],
        loc='upper center',
        bbox_to_anchor=(0.5, 0.90),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"  Saved: {save_path}")
    if show:
        plt.show()

    return fig