import matplotlib.pyplot as plt

# Plot images from the zip file
def plot_images(images, title):
    fig, axes = plt.subplots(5, 10, figsize=(10, 5))
    fig.suptitle(title, fontsize=16)
    axes = axes.flatten()
    for img, ax in zip(images, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def plot_images_from_zip(images):
    # Plot images from 'anastasia'
    plot_images(images['anastasia'], 'Anastasia Images')
    # Plot images from 'takao'
    plot_images(images['takao'], 'Takao Images')