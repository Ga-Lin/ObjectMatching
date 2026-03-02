import os
import cv2
import matplotlib.pyplot as plt

def load_and_resize_image(image_path, size=(100, 100)):
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR to RGB
        return cv2.resize(image, size)
    return None

def plot_rank_list(rank_list_path):

    with open(rank_list_path, 'r') as rank_list:
        rank_lines = rank_list.readlines()

    fig, axes = plt.subplots(50, 11, figsize=(22, 100)) # 50 rows, 11 columns

    # Loop through each query and its top 10 matches
    for row, line in enumerate(rank_lines):
        parts = line.strip().split(':')
        query_image_file = parts[0][1:]
        top_images = parts[1].strip().split()

        # Load and display the query image
        query_image_path = os.path.join(query_dir, query_image_file + '.jpg')
        query_image = load_and_resize_image(query_image_path)

        if query_image is not None:
            axes[row, 0].imshow(query_image)
            axes[row, 0].set_title(query_image_file)
        else:
            axes[row, 0].text(0.5, 0.5, 'Not Found', ha='center', va='center', fontsize=12)

        axes[row, 0].axis('off')

        # Load and display the top 10 images
        for col, gallery_idx in enumerate(top_images):
            gallery_image_path = os.path.join(gallery_dir, gallery_idx + '.jpg')
            gallery_image = load_and_resize_image(gallery_image_path)

            if gallery_image is not None:
                axes[row, col + 1].imshow(gallery_image)
                axes[row, col + 1].set_title(gallery_idx)
            else:
                axes[row, col + 1].text(0.5, 0.5, 'Not Found', ha='center', va='center', fontsize=12)

            axes[row, col + 1].axis('off')

    plt.tight_layout()
    plt.savefig(rank_list_path[2:].split('.')[0])


query_dir = './query'
gallery_dir = './gallery'
rank_list_resnet50_path = './rank_list_resnet50.txt'
rank_list_orb_path = './rank_list_orb.txt'

plot_rank_list(rank_list_resnet50_path)
plot_rank_list(rank_list_orb_path)