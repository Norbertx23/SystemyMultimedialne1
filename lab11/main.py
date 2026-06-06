import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

size_p = 512


def water_mark(img, mask, alpha=0.25):
    assert (img.shape[0] == mask.shape[0]) and (img.shape[1] == mask.shape[1]), "Wrong size"
    if len(img.shape) < 3:
        flag = True
        t_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    else:
        flag = False
        t_img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

    if (mask.dtype == bool):
        t_mask = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGBA)
    elif (mask.dtype == np.uint8):
        if len(mask.shape) < 3:
            t_mask = cv2.cvtColor((mask).astype(np.uint8), cv2.COLOR_GRAY2RGBA)
        else:
            t_mask = cv2.cvtColor((mask).astype(np.uint8), cv2.COLOR_RGB2RGBA)
    else:
        if len(mask.shape) < 3:
            t_mask = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGBA)
        else:
            t_mask = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_RGB2RGBA)

    t_out = cv2.addWeighted(t_img, 1, t_mask, alpha, 0)

    if flag:
        out = cv2.cvtColor(t_out, cv2.COLOR_RGBA2GRAY)
    else:
        out = cv2.cvtColor(t_out, cv2.COLOR_RGBA2RGB)
    return out


def put_data(img, data, binary_mask=np.uint8(1)):
    assert img.dtype == np.uint8, "img wrong data type"
    assert binary_mask.dtype == np.uint8, "binary_mask wrong data type"
    un_binary_mask = np.unpackbits(binary_mask)
    if data.dtype != bool:
        unpacked_data = np.unpackbits(data)
    else:
        unpacked_data = data
    dataspace = img.shape[0] * img.shape[1] * np.sum(un_binary_mask)
    assert (dataspace >= unpacked_data.size), "too much data"
    if dataspace == unpacked_data.size:
        prepered_data = unpacked_data.reshape(img.shape[0], img.shape[1], np.sum(un_binary_mask)).astype(np.uint8)
    else:
        prepered_data = np.resize(unpacked_data, (img.shape[0], img.shape[1], np.sum(un_binary_mask))).astype(np.uint8)
    mask = np.full((img.shape[0], img.shape[1]), binary_mask)
    img = np.bitwise_and(img, np.invert(mask))
    bv = 0
    for i, b in enumerate(un_binary_mask[::-1]):
        if b:
            temp = prepered_data[:, :, bv]
            temp = np.left_shift(temp, i)
            img = np.bitwise_or(img, temp)
            bv += 1
    return img


def pop_data(img, binary_mask=np.uint8(1), out_shape=None):
    un_binary_mask = np.unpackbits(binary_mask)
    data = np.zeros((img.shape[0], img.shape[1], np.sum(un_binary_mask))).astype(np.uint8)
    bv = 0
    for i, b in enumerate(un_binary_mask[::-1]):
        if b:
            mask = np.full((img.shape[0], img.shape[1]), 2 ** i)
            temp = np.bitwise_and(img, mask)
            data[:, :, bv] = temp[:, :].astype(np.uint8)
            bv += 1
    if out_shape != None:
        tmp = np.packbits(data.flatten())
        tmp = tmp[:np.prod(out_shape)]
        data = tmp.reshape(out_shape)
    return data


def calculate_metrics(original, modified):
    psnr_val = cv2.PSNR(original, modified)
    if len(original.shape) == 3:
        ssim_val = ssim(original, modified, channel_axis=-1)
    else:
        ssim_val = ssim(original, modified)
    return psnr_val, ssim_val


if __name__ == "__main__":

    carrier_img = cv2.imread('image0.jpg')
    secret_img = cv2.imread('image1.jpg')
    watermark_mask = cv2.imread('image_binary_0.png', cv2.IMREAD_GRAYSCALE)

    carrier_img = cv2.resize(carrier_img, (size_p, size_p))
    secret_img = cv2.resize(secret_img, (size_p, size_p))
    watermark_mask = cv2.resize(watermark_mask, (size_p, size_p))
    _, watermark_mask = cv2.threshold(watermark_mask, 127, 255, cv2.THRESH_BINARY)

    try:
        with open('cytat.txt', 'r', encoding='utf-8') as f:
            cytat = f.read()
    except FileNotFoundError:
        raise Exception("Wystąpił błąd")




    S_array = np.array([ord(c) for c in list(cytat)], dtype=np.uint8)
    cytat_bytes = np.frombuffer(S_array.tobytes(), dtype=np.uint8)

    B, G, R = cv2.split(carrier_img)
    B_steg = put_data(B.copy(), cytat_bytes, binary_mask=np.uint8(1))

    recovered_bytes = pop_data(B_steg, binary_mask=np.uint8(1), out_shape=cytat_bytes.shape)
    recovered_text = "".join([chr(b) for b in recovered_bytes])

    psnr_t, ssim_t = calculate_metrics(B, B_steg)
    print(f"[Tekst LSB] Kanał B -> PSNR: {psnr_t:.4f} dB | SSIM: {ssim_t:.4f}")
    print(f"Odzyskany tekst: {recovered_text[:50]}... (Poprawność: {cytat == recovered_text})")

    carrier_rgb = cv2.cvtColor(carrier_img, cv2.COLOR_BGR2RGB)
    steg_text_img = cv2.merge([B_steg, G, R])
    steg_text_rgb = cv2.cvtColor(steg_text_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 4))
    plt.suptitle("Ukrywanie tekstu w LSB kanału niebieskiego", fontsize=14)
    plt.subplot(1, 3, 1)
    plt.title("Oryginalny Nosiciel")
    plt.imshow(carrier_rgb)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Zakodowany LSB Blue")
    plt.imshow(steg_text_rgb)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Wizualizacja zmian w kanale B (Różnica x100)")
    plt.imshow(cv2.absdiff(B, B_steg) * 100, cmap='gray')
    plt.axis('off')
    plt.show()


    cap_B = (size_p * size_p * 3) // 8
    cap_G = (size_p * size_p * 2) // 8
    cap_R = (size_p * size_p * 2) // 8
    total_capacity = cap_B + cap_G + cap_R

    side_len = int(np.sqrt(total_capacity / 3))
    secret_resized = cv2.resize(secret_img, (side_len, side_len))
    sec_bytes = secret_resized.flatten()

    padded_sec = np.zeros(total_capacity, dtype=np.uint8)
    padded_sec[:sec_bytes.size] = sec_bytes

    data_B = padded_sec[:cap_B]
    data_G = padded_sec[cap_B:cap_B + cap_G]
    data_R = padded_sec[cap_B + cap_G:]

    B_img_steg = put_data(B.copy(), data_B, binary_mask=np.uint8(7))
    G_img_steg = put_data(G.copy(), data_G, binary_mask=np.uint8(3))
    R_img_steg = put_data(R.copy(), data_R, binary_mask=np.uint8(3))
    carrier_steg_full = cv2.merge([B_img_steg, G_img_steg, R_img_steg])

    psnr_b, ssim_b = calculate_metrics(B, B_img_steg)
    psnr_g, ssim_g = calculate_metrics(G, G_img_steg)
    psnr_r, ssim_r = calculate_metrics(R, R_img_steg)
    print(f"[Obraz LSB] Kanał B -> PSNR: {psnr_b:.4f} dB | SSIM: {ssim_b:.4f}")
    print(f"[Obraz LSB] Kanał G -> PSNR: {psnr_g:.4f} dB | SSIM: {ssim_g:.4f}")
    print(f"[Obraz LSB] Kanał R -> PSNR: {psnr_r:.4f} dB | SSIM: {ssim_r:.4f}")

    rec_data_B = pop_data(B_img_steg, binary_mask=np.uint8(7))
    rec_data_G = pop_data(G_img_steg, binary_mask=np.uint8(3))
    rec_data_R = pop_data(R_img_steg, binary_mask=np.uint8(3))

    rec_bits = np.concatenate([rec_data_B.flatten(), rec_data_G.flatten(), rec_data_R.flatten()])
    rec_bits_bool = rec_bits > 0
    rec_bytes = np.packbits(rec_bits_bool)

    rec_flat = rec_bytes[:secret_resized.size]
    recovered_secret = rec_flat.reshape(secret_resized.shape)

    steg_img_rgb = cv2.cvtColor(carrier_steg_full, cv2.COLOR_BGR2RGB)
    secret_rgb = cv2.cvtColor(recovered_secret, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(14, 6))

    title_text = (
        "Ukrywanie 2 obrazów\n"
        f"Jakość nosiciela -> Kanał B: PSNR={psnr_b:.2f}dB, SSIM={ssim_b:.4f} | "
        f"Kanał G: PSNR={psnr_g:.2f}dB, SSIM={ssim_g:.4f} | "
        f"Kanał R: PSNR={psnr_r:.2f}dB, SSIM={ssim_r:.4f}"
    )
    fig.suptitle(title_text, fontsize=14)

    axes[0].imshow(carrier_rgb)
    axes[0].set_title("Oryginał (Przed)")

    axes[1].imshow(steg_img_rgb)
    axes[1].set_title("Nosiciel z danymi (Po)")

    axes[2].imshow(secret_rgb)
    axes[2].set_title("Odzyskany ukryty obraz")

    for ax in axes:
        ax.set_xticks(range(0, 513, 100))
        ax.set_yticks(range(0, 513, 100))
    plt.tight_layout()
    plt.show()


    fig3, axes3 = plt.subplots(3, 4, figsize=(18, 13))
    fig3.suptitle("Deconstruction of Image (Bit Planes)", fontsize=20, fontweight='bold')

    gray_carrier = cv2.cvtColor(carrier_img, cv2.COLOR_BGR2GRAY)

    axes3_flat = axes3.flatten()

    axes3_flat[0].imshow(gray_carrier, cmap='gray')
    axes3_flat[0].set_title("Original Image", fontsize=13, fontweight='bold')
    axes3_flat[0].axis('off')

    bit_powers = [128, 64, 32, 16, 8, 4, 2, 1]
    for idx, power in enumerate(bit_powers):
        bit_plane = cv2.bitwise_and(gray_carrier, power)
        bit_plane[bit_plane > 0] = 255
        axes3_flat[idx + 1].imshow(bit_plane, cmap='gray')
        axes3_flat[idx + 1].set_title(f"bits -> Image & {power}", fontsize=12)
        axes3_flat[idx + 1].axis('off')

    recon_high = cv2.bitwise_and(gray_carrier, 128 + 64 + 32)
    recon_low = cv2.bitwise_and(gray_carrier, 8 + 4 + 2 + 1)

    axes3_flat[9].imshow(recon_high, cmap='gray')
    axes3_flat[9].set_title("Reconstructed [128,64,32]", fontsize=12)
    axes3_flat[9].axis('off')

    axes3_flat[10].imshow(recon_low, cmap='gray')
    axes3_flat[10].set_title("Reconstructed [8,4,2,1]", fontsize=12)
    axes3_flat[10].axis('off')

    axes3_flat[11].axis('off')

    plt.tight_layout()
    plt.show()


    alphas = [0.10, 0.25, 0.50]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    fig.suptitle("Analiza nakładania znaku wodnego", fontsize=14)

    axes[0].imshow(carrier_rgb)
    axes[0].set_title("Oryginał")
    axes[0].axis('off')

    for idx, alpha in enumerate(alphas):
        watermarked = water_mark(carrier_img, watermark_mask, alpha)
        watermarked_rgb = cv2.cvtColor(watermarked, cv2.COLOR_BGR2RGB)

        p, s = calculate_metrics(carrier_img, watermarked)
        print(f"[Watermark] alpha={alpha} -> PSNR: {p:.4f} dB | SSIM: {s:.4f}")

        axes[idx + 1].imshow(watermarked_rgb)
        axes[idx + 1].set_title(f"Watermark (alpha={alpha})\nPSNR: {p:.2f}dB")
        axes[idx + 1].axis('off')

        cv2.imwrite(f"watermark_alpha_{int(alpha * 100)}.png", watermarked)

    plt.tight_layout()
    plt.show()
