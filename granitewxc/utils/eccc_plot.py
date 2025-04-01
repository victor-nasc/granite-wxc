import time
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from granitewxc.utils.plot import plot_spatial


def plot_model_results(samples, samples_id, title, cbar_title, **kwargs):
    
    fig, axes = plt.subplots(1, len(samples), figsize=(18, 4))
    plt.suptitle(title, fontsize=15)
    
    for i, ax in enumerate(axes):
        im = plot_spatial(samples[i], ax, samples_id[i], **kwargs)
    
    # Aumentar a largura da barra de cores
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.05])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal') 
    cbar.set_label(cbar_title)  
    cbar.ax.tick_params()  

    plt.show()


def plot_power_spectrum(img, ax=None, label=None, save_fig=False):
    """
    A power spectrum measures the strength of features at different resolutions.

    :param img: H x W array.
    :param ax: Optional matplotlib Axes object to plot on.
    :param label: Optional label for the plot.
    :param save_fig: If True, saves the figure.
    """
    npix = img.shape[-2], img.shape[-1]

    fft_img = np.fft.fftn(img)
    fft_amp = np.abs(fft_img) ** 2
    fft_amp = fft_amp.flatten()

    kfreq_x = np.fft.fftfreq(npix[1]) * npix[1]  # wave vector
    kfreq_y = np.fft.fftfreq(npix[0]) * npix[0]  # wave vector
    kfreq2D = np.meshgrid(kfreq_x, kfreq_y)
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2).flatten()

    kbins = np.arange(0.5, min(*npix) // 2 + 1, 1.0)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(
        knrm, fft_amp, statistic='mean', bins=kbins
    )
    Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)

    # If ax is provided, plot on that axis; otherwise, create a new plot.
    if ax is None:
        fig, ax = plt.subplots()
    ax.loglog(kvals, Abins, label=label)
    ax.set_xlabel("$k$")
    ax.set_ylabel("$P(k)$")
    if label:
        ax.legend()
    plt.tight_layout()

    if save_fig:
        timestr = time.strftime("%Y%m%d-%H%M")
        plt.savefig(f'power_spectrum_{timestr}.png', dpi=300, bbox_inches='tight')

    return ax


def plot_residual_and_power_spectrum(residual, target, prediction, **kwargs):

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    im_residual = plot_spatial(residual, axes[0], "Residual", **kwargs.get('plot_residual_kwargs', {}))
    fig.colorbar(im_residual, ax=axes[0], orientation='horizontal', fraction=0.05, aspect=32)
    
    plot_power_spectrum(target, ax=axes[1], label='Target')
    plot_power_spectrum(prediction, ax=axes[1], label='AI Model')
    
    axes[1].set_title("Power Spectrum")
    axes[1].legend(['Target', 'AI Model'], fontsize=12)
    axes[1].set_xlabel("Frequency")
    axes[1].set_ylabel("Power")
    axes[1].tick_params(labelsize=8)
    
    plt.show()


def plot_sample(data):

    x = data['x']
    y = data['y']

    num_rows, num_cols = 2, 2
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(6, 6))
    
    # flatten the axes for easy iteration
    axs = np.ravel(axs)
    
    images = [
        axs[0].imshow(x[0, 1, :, :], cmap='coolwarm'), axs[1].imshow(y[0, 0, :, :], cmap='coolwarm'), 
        axs[2].imshow(x[0, 2, :, :], cmap='coolwarm'), axs[3].imshow(y[0, 1, :, :], cmap='coolwarm')
    ]
    
    titles = ["GDPS - UUVE", "HRDPS - UUVE", "GDPS - VVSN", "HRDPS - VVSN"]
    for ax, title in zip(axs, titles):
        ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=14, pad=10)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])  
    
    # colorbar
    for i in range(num_rows):
        cbar_ax = fig.add_axes([0.92, [0.56, 0.073][i], 0.02, 0.35])
        fig.colorbar(axs[i * num_cols].images[0], cax=cbar_ax, orientation='vertical', label=["UUVE [m/s]", "VVSN [m/s]"][i])

    plt.show()


def plot_loss(train_loss, val_loss):
    
    plt.figure(figsize=(7, 4)) 
    plt.plot(train_loss, label='Training Loss', color='blue', linestyle='-', linewidth=1)
    plt.plot(val_loss, label='Validation Loss', color='orange', linestyle='-', linewidth=1)

    plt.title('Training and Validation Loss', fontsize=13)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.show()