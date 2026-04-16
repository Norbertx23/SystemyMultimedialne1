import numpy as np

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
    while i < len(flat_data):
        count = _search_for_repetitions(flat_data[i:])
        buffer[idx_buffer] = count
        buffer[idx_buffer + 1] = flat_data[i]
        idx_buffer += 2
        i += count

    return buffer[:idx_buffer]


def rle_decoder(stream: np.array) -> np.array:
    num_dims = stream[0]

    shape = tuple(stream[1:num_dims + 1].astype(int))

    encoded_data = stream[num_dims + 1:]

    counts = encoded_data[0::2]
    values = encoded_data[1::2]

    decoded_flat = np.repeat(values, counts)

    return decoded_flat.reshape(shape)

def byte_run_encoder(np_array: np.array) -> np.array:
    # Implementacja kodowania Byte Run
    pass
def byte_decoder(stream: np.array) -> np.array:
    # Implementacja dekodowania Byte Run
    pass


a = np.array([[1, 1, 1, 2, 2], [3, 3, 4, 4, 4]])
encoded = rle_encoder(a)
print("Encoded:", encoded)
decoded = rle_decoder(encoded)
print("Decoded:\n", decoded)
