import numpy as np
from exploring_exploration.cython_extensions.tsp import exploration_tsp
from exploring_exploration.cython_extensions.oe import exploration_entropy
from matplotlib import pyplot as plt

if __name__ == '__main__':
    dim = 2  # Dimension of the points
    sequence_length = 100  # Number of points to generate

    # Generate 100 random 2D points. Has to be float32 for the Cython extension.
    sample_points = np.random.rand(sequence_length, dim).astype(np.float32)
    # calculate otsd for sample_points
    otsd = exploration_tsp(sample_points)

    bounds = 2 * np.sqrt(5 * dim) * ((3 / 2) * np.arange(1, sequence_length + 1)) ** (1 - 1 / dim)

    # normalize otsd
    otsd_norm = otsd / bounds

    # compute oe
    oe = exploration_entropy(sample_points)

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    # plot otsd, otsd_norm, oe
    for ax, data, title in zip(axs, [otsd, otsd_norm, oe], ['OTSD', 'Normalized OTSD', 'OE']):
        ax.plot(data, marker='o', markevery=10)
        ax.set_title(title)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value')
        ax.grid(True)
    plt.tight_layout()
    plt.show()

