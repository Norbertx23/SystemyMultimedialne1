import numpy as np
import sys
import cv2
from tqdm import tqdm

def _search_for_repetitions(np_array: np.array) -> np.array:
    if len(np_array) == 0:
        return 0

    counter = 1
    for i in range(1, len(np_array)):
        if np_array[i] != np_array[i-1]:
            break
        counter+=1
    return counter

def _search_for_non_repetitions(np_array: np.array) -> np.array:
    if len(np_array) == 0:
        return 0

    counter = 1
    for i in range(1, len(np_array)):
        if np_array[i] == np_array[i-1]:
            counter -= 1
            break
        counter+=1
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
    with tqdm(total=len(flat_data), desc="Kompresja RLE") as pbar:
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
    with tqdm(total=len(encoded_data), desc="Dekompresja RLE") as pbar:
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
    with tqdm(total=len(flat_data), desc="Kompresja ByteRun") as pbar:
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
                    buffer[idx_out + 1 : idx_out + 1 + 128] = flat_data[i + offset : i + offset + 128]
                    idx_out += (1 + 128)
                    offset += 128
                    counter -= 128

                if counter > 0:
                    buffer[idx_out] = counter - 1
                    buffer[idx_out + 1 : idx_out + 1 + counter] = flat_data[i + offset : i + offset + counter]
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
    with tqdm(total=len(data), desc="Dekompresja ByteRun") as pbar:
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

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, np.ndarray):
        size = obj.nbytes
    elif isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def test_image_compression(image_path):
    print(f"\n{'=' * 50}")
    print(f"ANALIZA OBRAZU: {image_path}")
    print(f"{'=' * 50}")


    img = cv2.imread(image_path)
    img_int = img.astype(int)

    original_size_bytes = get_size(img_int)

    print(f"Kształt oryginału: {img_int.shape}")
    print(f"Rozmiar oryginału w pamięci: {original_size_bytes} bajtów\n")

    print("--- METODA: RLE ---")
    encoded_rle = rle_encoder(img_int)
    decoded_rle = rle_decoder(encoded_rle)

    rle_size_bytes = get_size(encoded_rle)
    cr_rle = original_size_bytes / rle_size_bytes
    percent_rle = (rle_size_bytes / original_size_bytes) * 100
    is_identical_rle = np.array_equal(img_int, decoded_rle)

    print(f"Rozmiar po kompresji: {rle_size_bytes} bajtów")
    print(f"Czy dane identyczne z oryginałem? {'TAK' if is_identical_rle else 'NIE'}")
    print(f"{cr_rle=:.2f}")
    print(f"{percent_rle=:.2f}%\n")

    print("--- METODA: BYTERUN ---")
    encoded_br = byte_run_encoder(img_int)
    decoded_br = byte_decoder(encoded_br)

    br_size_bytes = get_size(encoded_br)
    cr_br = original_size_bytes / br_size_bytes
    percent_br = (br_size_bytes / original_size_bytes) * 100
    is_identical_br = np.array_equal(img_int, decoded_br)

    print(f"Rozmiar po kompresji: {br_size_bytes} bajtów")
    print(f"Czy dane identyczne z oryginałem? {'TAK' if is_identical_br else 'NIE'}")
    print(f"{cr_br=:.2f}")
    print(f"{percent_br=:.2f}%")

if __name__ == "__main__":
    test_cases = ('Color.png',
                  'doc.png',
                  'tech.png')

    for test_case in test_cases:
        test_image_compression(test_case)
