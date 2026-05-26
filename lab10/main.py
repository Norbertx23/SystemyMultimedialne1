import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from skimage.metrics import structural_similarity as ssim


def JPG_comp(photo, quality_params):
    img = cv2.imread(photo)
    num_plots = len(quality_params) + 1
    fig, axs = plt.subplots(1, num_plots, figsize=(22, 5), num=f"JPG - {photo}")
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
    plt.close(fig)

def Blur_comp(photo, kernel_sizes):
    img = cv2.imread(photo)
    num_plots = len(kernel_sizes) + 1
    fig, axs = plt.subplots(1, num_plots, figsize=(22, 5), num=f"Blur - {photo}")
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Oryginal")
    axs[0].axis('off')
    output_dir = 'blur'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, k in enumerate(kernel_sizes):
        blur_img = cv2.blur(img, (k, k))
        axs[i + 1].imshow(cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB))
        axs[i + 1].set_title(f"Blur: {k}x{k}")
        axs[i + 1].axis('off')
        base_name = photo.split('.')[0]
        out_path = os.path.join(output_dir, f'{base_name}_blur_{k}.png')
        cv2.imwrite(out_path, blur_img)
    plt.tight_layout()
    plt.close(fig)

def Noise_comp(photo, alpha_params, sigma=50):
    img = cv2.imread(photo)
    num_plots = len(alpha_params) + 1
    fig, axs = plt.subplots(1, num_plots, figsize=(22, 5), num=f"Noise - {photo}")
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
    plt.close(fig)

def calculate_mse(I, K):
    I = I.astype(np.float64)
    K = K.astype(np.float64)
    return np.mean((I - K) ** 2)

def calculate_nmse(I, K):
    I = I.astype(np.float64)
    K = K.astype(np.float64)
    denominator = np.mean(K ** 2)
    if denominator == 0:
        return float('inf')
    return np.mean((I - K) ** 2) / denominator

def calculate_psnr(I, K):
    mse_val = calculate_mse(I, K)
    if mse_val == 0:
        return float('inf')
    max_i = 255.0
    return 10 * np.log10((max_i ** 2) / mse_val)

def calculate_if(I, K):
    I = I.astype(np.float64)
    K = K.astype(np.float64)
    numerator = np.sum((K - I) ** 2)
    denominator = np.sum(I ** 2)
    if denominator == 0:
        return float('-inf')
    return 1.0 - (numerator / denominator)

def calculate_ssim(I, K):
    return ssim(I, K, channel_axis=2, data_range=255)

def evaluate_metrics(photos, configs):
    results = []
    for photo in photos:
        if not os.path.exists(photo):
            continue
        img_orig = cv2.imread(photo)
        base_name = photo.split('.')[0]
        for method, config in configs.items():
            for p in config['params']:
                filepath = os.path.join(config['dir'], f"{base_name}_{config['prefix']}_{p}.png")
                if os.path.exists(filepath):
                    img_deg = cv2.imread(filepath)
                    results.append({
                        'Zdjecie': base_name,
                        'Metoda': method,
                        'Parametr': p,
                        'MSE': calculate_mse(img_orig, img_deg),
                        'NMSE': calculate_nmse(img_orig, img_deg),
                        'PSNR': calculate_psnr(img_orig, img_deg),
                        'IF': calculate_if(img_orig, img_deg),
                        'SSIM': calculate_ssim(img_orig, img_deg)
                    })
    return pd.DataFrame(results)

def display_tables_and_plots(df, configs):
    pd.options.display.float_format = '{:.4f}'.format
    metrics = ['MSE', 'NMSE', 'PSNR', 'IF', 'SSIM']
    with pd.ExcelWriter('wyniki_miar.xlsx', engine='openpyxl') as writer:
        for metric in metrics:
            print(f"\n--- TABELA DLA MIARY: {metric} ---")
            pivot = pd.pivot_table(df, values=metric, index=['Metoda', 'Parametr'], columns=['Zdjecie'])
            print(pivot)
            pivot.to_excel(writer, sheet_name=metric)

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle('Korelacja miar obiektywnych z poziomem znieksztalcen', fontsize=16)

    for i, method in enumerate(configs.keys()):
        df_method = df[df['Metoda'] == method]
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            for photo_name in df_method['Zdjecie'].unique():
                df_photo = df_method[df_method['Zdjecie'] == photo_name].sort_values(by='Parametr')
                ax.plot(df_photo['Parametr'], df_photo[metric], marker='o', label=photo_name)

            if i == 0:
                ax.set_title(metric)
            if j == 0:
                ax.set_ylabel(method)

            if method == 'JPEG':
                ax.invert_xaxis()

            ax.grid(True, linestyle='--', alpha=0.7)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    plt.show()

def test_function():
    photos = ['corgie.jpg', 'gory.jpg', 'kaktusy.jpg']

    configs = {
        'JPEG': {'params': [95, 85, 75, 65, 55, 45, 35, 25, 15, 5], 'dir': 'jpeg', 'prefix': 'jpeg'},
        'Blur': {'params': [3, 5, 7, 9, 13, 17, 21, 25, 31, 35], 'dir': 'blur', 'prefix': 'blur'},
        'Noise': {'params': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'dir': 'noise', 'prefix': 'noise'}
    }

    for photo in photos:
        if os.path.exists(photo):
            JPG_comp(photo, configs['JPEG']['params'])
            Blur_comp(photo, configs['Blur']['params'])
            Noise_comp(photo, configs['Noise']['params'])
        else:
            print(f"Brak pliku: {photo}")

    df_results = evaluate_metrics(photos, configs)

    if not df_results.empty:
        display_tables_and_plots(df_results, configs)

if __name__ == "__main__":
    test_function()