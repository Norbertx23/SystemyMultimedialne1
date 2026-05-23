import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

photos = ['corgie.jpg', 'gory.jpg', 'kaktusy.jpg']

def JPG_comp(photo):
    img = cv2.imread(photo)

    quality_params = [80, 50, 15]
    num_plots = len(quality_params) + 1

    fig, axs = plt.subplots(1, num_plots, figsize=(15, 5),num="JPG")

    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Oryginal")
    axs[0].axis('off')

    output_dir = 'jpeg'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, q in enumerate(quality_params):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]

        result, encimg = cv2.imencode('.jpg', img, encode_param)
        decimg = cv2.imdecode(encimg, 1)

        axs[i + 1].imshow(cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB))
        axs[i + 1].set_title(f"JPEG: {q}")
        axs[i + 1].axis('off')

        base_name = photo.split('.')[0]
        out_path = os.path.join(output_dir, f'{base_name}_jpeg_{q}.png')
        cv2.imwrite(out_path, decimg)

    plt.tight_layout()
    plt.show()

def Blur_comp(photo):
    img = cv2.imread(photo)

    kernel_sizes = [(5, 5), (15, 15), (35, 35)]
    num_plots = len(kernel_sizes) + 1

    fig, axs = plt.subplots(1, num_plots, figsize=(15, 5),num="Blur")

    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Oryginal")
    axs[0].axis('off')

    output_dir = 'blur'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, k in enumerate(kernel_sizes):
        blur_img = cv2.blur(img, k)

        axs[i + 1].imshow(cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB))
        axs[i + 1].set_title(f"Blur: {k[0]}x{k[1]}")
        axs[i + 1].axis('off')

        base_name = photo.split('.')[0]
        out_path = os.path.join(output_dir, f'{base_name}_blur_{k[0]}.png')
        cv2.imwrite(out_path, blur_img)

    plt.tight_layout()
    plt.show()

def Noise_comp(photo):
    img = cv2.imread(photo)

    alpha_params = [0.2, 0.6, 1.0]
    sigma = 50
    num_plots = len(alpha_params) + 1

    fig, axs = plt.subplots(1, num_plots, figsize=(15, 5),num="Noise")

    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Oryginal")
    axs[0].axis('off')

    output_dir = 'noise'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gauss = np.random.normal(0, sigma, img.shape)

    for i, alpha in enumerate(alpha_params):
        noisy_img = (img + alpha * gauss).clip(0, 255).astype(np.uint8)

        axs[i + 1].imshow(cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB))
        axs[i + 1].set_title(f"Szum alpha: {alpha}")
        axs[i + 1].axis('off')

        base_name = photo.split('.')[0]
        out_path = os.path.join(output_dir, f'{base_name}_noise_{alpha}.png')
        cv2.imwrite(out_path, noisy_img)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    for photo in photos:
        JPG_comp(photo)
        Blur_comp(photo)
        Noise_comp(photo)