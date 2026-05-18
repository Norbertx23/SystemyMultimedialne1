import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

##############################################################################
######   Konfiguracja       ##################################################
##############################################################################

kat = r'.'  # katalog z plikami wideo
plik = "clip_4.mp4"  # nazwa pliku
ile = 20
key_frame_counter = 20  # Klatka kluczowa co 4 klatki
# plot_frames = np.array([15, 31, 47])
plot_frames = np.array([3, 7, 11,])
auto_pause_frames = np.array([])  # Bez auto-pauzy
subsampling = "4:2:0"  # ZMIENIAJ W TRAKCIE TESTÓW (np. 4:2:2, 4:2:0, 4:1:0)
dzielnik = 4  # ZMIENIAJ W TRAKCIE TESTÓW (np. 2, 4, 8)
wyswietlaj_kaltki = False  # Pokazuj podgląd wideo
# ROI = [[200, 500, 300, 800]]            # Środek kadru - łapie idącego mężczyznę
ROI = [[300, 600, 500, 900]]
metoda_kompresji_strumieniowej = 'RLE'  # 'brak', 'RLE' lub 'ByteRun'


##############################################################################
####     Kompresja strumieniowa     ##########################################
##############################################################################

def _search_for_repetitions(np_array: np.array) -> int:
    if len(np_array) == 0:
        return 0
    counter = 1
    for i in range(1, len(np_array)):
        if np_array[i] != np_array[i - 1]:
            break
        counter += 1
    return counter


def _search_for_non_repetitions(np_array: np.array) -> int:
    if len(np_array) == 0:
        return 0
    counter = 1
    for i in range(1, len(np_array)):
        if np_array[i] == np_array[i - 1]:
            counter -= 1
            break
        counter += 1
    return counter


def rle_encoder(np_array: np.array) -> np.array:
    shape = np_array.shape
    header = np.array([len(shape)], dtype=int)
    header = np.concatenate([header, shape])
    flat_data = np_array.flatten()
    if len(flat_data) == 0:
        return header
    max_size = len(flat_data) * 2
    buffer = np.zeros(len(header) + max_size, dtype=int)
    buffer[:len(header)] = header
    idx_buffer = len(header)
    i = 0
    with tqdm(total=len(flat_data), desc="Kompresja RLE", leave=False) as pbar:
        while i < len(flat_data):
            count = _search_for_repetitions(flat_data[i:])
            buffer[idx_buffer] = count
            buffer[idx_buffer + 1] = flat_data[i]
            idx_buffer += 2
            i += count
            pbar.update(count)
    return buffer[:idx_buffer]


def rle_decoder(stream: np.array) -> np.array:
    num_dims = stream[0]
    shape = tuple(stream[1: num_dims + 1].astype(int))
    encoded_data = stream[num_dims + 1:]
    total_pixels = np.prod(shape)
    decoded_buffer = np.zeros(total_pixels, dtype=int)
    idx_out = 0
    i = 0
    with tqdm(total=len(encoded_data), desc="Dekompresja RLE", leave=False) as pbar:
        while i < len(encoded_data):
            count = int(encoded_data[i])
            value = encoded_data[i + 1]
            decoded_buffer[idx_out: idx_out + count] = value
            idx_out += count
            i += 2
            pbar.update(2)
    return decoded_buffer.reshape(shape)


def byte_run_encoder(np_array: np.array) -> np.array:
    shape = np_array.shape
    header = np.array([len(shape)], dtype=int)
    header = np.concatenate([header, shape])
    flat_data = np_array.flatten()
    if len(flat_data) == 0:
        return header
    max_size = len(flat_data) * 2
    buffer = np.zeros(len(header) + max_size, dtype=int)
    buffer[:len(header)] = header
    idx_out = len(header)
    i = 0
    with tqdm(total=len(flat_data), desc="Kompresja ByteRun", leave=False) as pbar:
        while i < len(flat_data):
            if i + 1 < len(flat_data) and flat_data[i] == flat_data[i + 1]:
                counter = _search_for_repetitions(flat_data[i:])
                jump = counter
                while counter > 128:
                    buffer[idx_out] = -127
                    buffer[idx_out + 1] = flat_data[i]
                    idx_out += 2
                    counter -= 128
                if counter > 0:
                    buffer[idx_out] = -(counter - 1)
                    buffer[idx_out + 1] = flat_data[i]
                    idx_out += 2
                i += jump
            else:
                counter = _search_for_non_repetitions(flat_data[i:])
                jump = counter
                offset = 0
                while counter > 128:
                    buffer[idx_out] = 127
                    buffer[idx_out + 1: idx_out + 1 + 128] = flat_data[i + offset: i + offset + 128]
                    idx_out += (1 + 128)
                    offset += 128
                    counter -= 128
                if counter > 0:
                    buffer[idx_out] = counter - 1
                    buffer[idx_out + 1: idx_out + 1 + counter] = flat_data[i + offset: i + offset + counter]
                    idx_out += (1 + counter)
                i += jump
                pbar.update(jump)
    return buffer[:idx_out]


def byte_decoder(stream: np.array) -> np.array:
    num_dims = stream[0]
    shape = tuple(stream[1: num_dims + 1].astype(int))
    data = stream[num_dims + 1:]
    decoded = []
    idx = 0
    with tqdm(total=len(data), desc="Dekompresja ByteRun", leave=False) as pbar:
        while idx < len(data):
            start_idx = idx
            control = data[idx]
            idx += 1
            if control < 0:
                n = -control + 1
                value = data[idx]
                decoded.extend([value] * n)
                idx += 1
            else:
                n = control + 1
                decoded.extend(data[idx: idx + n])
                idx += n
            pbar.update(idx - start_idx)
    return np.array(decoded).reshape(shape)


##############################################################################
####     Kompresja i dekompresja wideo #######################################
##############################################################################
class data:
    def __init__(self):
        self.Y = None
        self.Cb = None
        self.Cr = None
        self.semi_Y = None
        self.semi_Cb = None
        self.semi_Cr = None


def Chroma_subsampling(L, subsampling):
    if subsampling == "4:4:4":
        return L
    elif subsampling == "4:2:2":
        return L[:, ::2]
    elif subsampling == "4:4:0":
        return L[::2, :]
    elif subsampling == "4:2:0":
        return L[::2, ::2]
    elif subsampling == "4:1:1":
        return L[:, ::4]
    elif subsampling == "4:1:0":
        return L[::2, ::4]
    return L


def Chroma_resampling(L, subsampling):
    if subsampling == "4:4:4":
        return L
    elif subsampling == "4:2:2":
        return np.repeat(L, 2, axis=1)
    elif subsampling == "4:4:0":
        return np.repeat(L, 2, axis=0)
    elif subsampling == "4:2:0":
        return np.repeat(np.repeat(L, 2, axis=1), 2, axis=0)
    elif subsampling == "4:1:1":
        return np.repeat(L, 4, axis=1)
    elif subsampling == "4:1:0":
        return np.repeat(np.repeat(L, 4, axis=1), 2, axis=0)
    return L


def frame_image_to_class(frame, subsampling):
    Frame_class = data()
    Frame_class.Y = frame[:, :, 0].astype(int)
    Frame_class.Cr = Chroma_subsampling(frame[:, :, 1].astype(int), subsampling)
    Frame_class.Cb = Chroma_subsampling(frame[:, :, 2].astype(int), subsampling)
    return Frame_class


def frame_layers_to_image(Y, Cr, Cb, subsampling):
    Cb = Chroma_resampling(Cb, subsampling)
    Cr = Chroma_resampling(Cr, subsampling)
    min_h = min(Y.shape[0], Cr.shape[0], Cb.shape[0])
    min_w = min(Y.shape[1], Cr.shape[1], Cb.shape[1])
    return np.dstack([Y[:min_h, :min_w], Cr[:min_h, :min_w], Cb[:min_h, :min_w]]).clip(0, 255).astype(np.uint8)


def compress_KeyFrame(Frame_class):
    KeyFrame = data()
    KeyFrame.semi_Y = Frame_class.Y.copy()
    KeyFrame.semi_Cb = Frame_class.Cb.copy()
    KeyFrame.semi_Cr = Frame_class.Cr.copy()
    if metoda_kompresji_strumieniowej == 'RLE':
        KeyFrame.Y = rle_encoder(KeyFrame.semi_Y)
        KeyFrame.Cb = rle_encoder(KeyFrame.semi_Cb)
        KeyFrame.Cr = rle_encoder(KeyFrame.semi_Cr)
    elif metoda_kompresji_strumieniowej == 'ByteRun':
        KeyFrame.Y = byte_run_encoder(KeyFrame.semi_Y)
        KeyFrame.Cb = byte_run_encoder(KeyFrame.semi_Cb)
        KeyFrame.Cr = byte_run_encoder(KeyFrame.semi_Cr)
    else:
        KeyFrame.Y = KeyFrame.semi_Y
        KeyFrame.Cb = KeyFrame.semi_Cb
        KeyFrame.Cr = KeyFrame.semi_Cr
    return KeyFrame


def decompress_KeyFrame(KeyFrame):
    if metoda_kompresji_strumieniowej == 'RLE':
        Y = rle_decoder(KeyFrame.Y)
        Cb = rle_decoder(KeyFrame.Cb)
        Cr = rle_decoder(KeyFrame.Cr)
    elif metoda_kompresji_strumieniowej == 'ByteRun':
        Y = byte_decoder(KeyFrame.Y)
        Cb = byte_decoder(KeyFrame.Cb)
        Cr = byte_decoder(KeyFrame.Cr)
    else:
        Y = KeyFrame.semi_Y
        Cb = KeyFrame.semi_Cb
        Cr = KeyFrame.semi_Cr
    return frame_layers_to_image(Y, Cr, Cb, subsampling)


def compress_not_KeyFrame(Frame_class, KeyFrame):
    Compress_data = data()
    Compress_data.semi_Y = Frame_class.Y.copy()
    Compress_data.semi_Cb = Frame_class.Cb.copy()
    Compress_data.semi_Cr = Frame_class.Cr.copy()
    diff_Y = (Frame_class.Y - KeyFrame.semi_Y) // dzielnik
    diff_Cb = (Frame_class.Cb - KeyFrame.semi_Cb) // dzielnik
    diff_Cr = (Frame_class.Cr - KeyFrame.semi_Cr) // dzielnik
    if metoda_kompresji_strumieniowej == 'RLE':
        Compress_data.Y = rle_encoder(diff_Y)
        Compress_data.Cb = rle_encoder(diff_Cb)
        Compress_data.Cr = rle_encoder(diff_Cr)
    elif metoda_kompresji_strumieniowej == 'ByteRun':
        Compress_data.Y = byte_run_encoder(diff_Y)
        Compress_data.Cb = byte_run_encoder(diff_Cb)
        Compress_data.Cr = byte_run_encoder(diff_Cr)
    else:
        Compress_data.Y = diff_Y
        Compress_data.Cb = diff_Cb
        Compress_data.Cr = diff_Cr
    return Compress_data


def decompress_not_KeyFrame(Compress_data, KeyFrame):
    if metoda_kompresji_strumieniowej == 'RLE':
        diff_Y = rle_decoder(Compress_data.Y)
        diff_Cb = rle_decoder(Compress_data.Cb)
        diff_Cr = rle_decoder(Compress_data.Cr)
    elif metoda_kompresji_strumieniowej == 'ByteRun':
        diff_Y = byte_decoder(Compress_data.Y)
        diff_Cb = byte_decoder(Compress_data.Cb)
        diff_Cr = byte_decoder(Compress_data.Cr)
    else:
        diff_Y = Compress_data.Y
        diff_Cb = Compress_data.Cb
        diff_Cr = Compress_data.Cr
    Y = KeyFrame.semi_Y + (diff_Y * dzielnik)
    Cb = KeyFrame.semi_Cb + (diff_Cb * dzielnik)
    Cr = KeyFrame.semi_Cr + (diff_Cr * dzielnik)
    return frame_layers_to_image(Y, Cr, Cb, subsampling)


##############################################################################
####     Głowna pętla programu      ##########################################
##############################################################################

cap = cv2.VideoCapture(os.path.join(kat, plik))
if ile < 0: ile = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if wyswietlaj_kaltki:
    cv2.namedWindow('Normal Frame')
    cv2.namedWindow('Decompressed Frame')

compression_information = np.zeros((3, ile))
zebrane_do_wykresu = []

for i in range(ile):
    print(f"\nPrzetwarzanie klatki {i + 1}/{ile}...")
    ret, frame = cap.read()
    if not ret: break

    if wyswietlaj_kaltki: cv2.imshow('Normal Frame', frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    Frame_class = frame_image_to_class(frame, subsampling)

    if (i % key_frame_counter) == 0:
        KeyFrame = compress_KeyFrame(Frame_class)
        cY = KeyFrame.Y
        cCb = KeyFrame.Cb
        cCr = KeyFrame.Cr
        Decompresed_Frame = decompress_KeyFrame(KeyFrame)
    else:
        Compress_data = compress_not_KeyFrame(Frame_class, KeyFrame)
        cY = Compress_data.Y
        cCb = Compress_data.Cb
        cCr = Compress_data.Cr
        Decompresed_Frame = decompress_not_KeyFrame(Compress_data, KeyFrame)

    compression_information[0, i] = (frame[:, :, 0].size - cY.size) / frame[:, :, 0].size
    compression_information[1, i] = (frame[:, :, 0].size - cCb.size) / frame[:, :, 0].size
    compression_information[2, i] = (frame[:, :, 0].size - cCr.size) / frame[:, :, 0].size

    if wyswietlaj_kaltki:
        cv2.imshow('Decompressed Frame', cv2.cvtColor(Decompresed_Frame, cv2.COLOR_YCrCb2BGR))

    if np.any(plot_frames == i):
        for r in ROI:
            ref_RGB = cv2.cvtColor(frame, cv2.COLOR_YCrCb2RGB)
            dec_RGB = cv2.cvtColor(Decompresed_Frame, cv2.COLOR_YCrCb2RGB)

            ref_crop = ref_RGB[r[0]:r[1], r[2]:r[3]]
            dec_crop = dec_RGB[r[0]:r[1], r[2]:r[3]]

            diff = np.abs(ref_crop.astype(float) - dec_crop.astype(float))
            diff_img = np.clip(diff, 0, 255).astype(np.uint8)

            ref_YCrCb = frame[r[0]:r[1], r[2]:r[3]]
            dec_YCrCb = Decompresed_Frame[r[0]:r[1], r[2]:r[3]]

            diff_Y = np.clip(np.abs(ref_YCrCb[:, :, 0].astype(float) - dec_YCrCb[:, :, 0].astype(float)), 0,
                             255).astype(np.uint8)
            diff_Cr = np.clip(np.abs(ref_YCrCb[:, :, 1].astype(float) - dec_YCrCb[:, :, 1].astype(float)), 0,
                              255).astype(np.uint8)
            diff_Cb = np.clip(np.abs(ref_YCrCb[:, :, 2].astype(float) - dec_YCrCb[:, :, 2].astype(float)), 0,
                              255).astype(np.uint8)

            zebrane_do_wykresu.append((ref_crop, diff_img, dec_crop, diff_Y, diff_Cb, diff_Cr, i))

    if np.any(auto_pause_frames == i): cv2.waitKey(-1)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('p'):
        cv2.waitKey(-1)

cap.release()
cv2.destroyAllWindows()

# Przygotowanie stringu do nazw plików
safe_sub = subsampling.replace(':', '')
nazwa_baza = f"{plik}_sub{safe_sub}_div{dzielnik}_{metoda_kompresji_strumieniowej}"

##############################################################################
####     Generowanie i zapis Wykresu Wizualnego (3 obrazki) ##################
##############################################################################
if len(zebrane_do_wykresu) > 0:
    fig, axs = plt.subplots(len(zebrane_do_wykresu)* 2, 3, figsize=(15, 4 * len(zebrane_do_wykresu)))

    # Obsługa przypadku, gdy wybraliśmy tylko 1 klatkę do narysowania
    if len(zebrane_do_wykresu) == 1:
        axs = [axs]

    for idx, (oryg, diff_rgb, zdek, diff_y, diff_cb, diff_cr, nr_klatki) in enumerate(zebrane_do_wykresu):
        row_idx = idx * 2
        axs[row_idx][0].imshow(oryg)
        axs[row_idx][0].set_title(f'Oryginał (Klatka {nr_klatki})')
        axs[row_idx][0].axis('off')

        axs[row_idx][1].imshow(diff_rgb)
        axs[row_idx][1].set_title(f'Różnica (Klatka {nr_klatki})')
        axs[row_idx][1].axis('off')

        axs[row_idx][2].imshow(zdek)
        axs[row_idx][2].set_title(f'Zdekodowana (Klatka {nr_klatki})')
        axs[row_idx][2].axis('off')

        axs[row_idx + 1][0].imshow(diff_y, cmap='gray')
        axs[row_idx + 1][0].set_title(f'Różnica Y (Luminancja)')
        axs[row_idx + 1][0].axis('off')

        axs[row_idx + 1][1].imshow(diff_cb, cmap='gray')
        axs[row_idx + 1][1].set_title(f'Różnica Cb (Chrominancja)')
        axs[row_idx + 1][1].axis('off')

        axs[row_idx + 1][2].imshow(diff_cr, cmap='gray')
        axs[row_idx + 1][2].set_title(f'Różnica Cr (Chrominancja)')
        axs[row_idx + 1][2].axis('off')

    plt.tight_layout()
    nazwa_wizualna = f"wizualizacja_{nazwa_baza}.png"
    plt.savefig(nazwa_wizualna)
    print(f"\nZapisano plik: {nazwa_wizualna}")
    plt.show()

##############################################################################
####     Generowanie i zapis Wykresu Liniowego (Kompresja)  ##################
##############################################################################
plt.figure()
plt.plot(np.arange(0, ile), compression_information[0, :] * 100, label='Kompresja Y (%)')
plt.plot(np.arange(0, ile), compression_information[1, :] * 100, label='Kompresja Cb (%)')
plt.plot(np.arange(0, ile), compression_information[2, :] * 100, label='Kompresja Cr (%)')
plt.title(f"Plik: {plik}, subsampling={subsampling}, div={dzielnik}, alg.={metoda_kompresji_strumieniowej}, {key_frame_counter=}")
plt.legend()
plt.xlabel("Numer klatki")
plt.ylabel("% zysku pamięci")

nazwa_wykres = f"wykres_kompresji_{nazwa_baza+str(key_frame_counter)}.png"
plt.savefig(nazwa_wykres)
print(f"Zapisano plik: {nazwa_wykres}")
plt.show()