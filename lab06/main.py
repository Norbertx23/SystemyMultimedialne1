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

    return buffer[:idx_out]

def byte_decoder(stream: np.array) -> np.array:
    num_dims = stream[0]
    shape = tuple(stream[1: num_dims + 1].astype(int))

    # Dane zaczynają się po nagłówku
    data = stream[num_dims + 1:]
    decoded = []

    idx = 0
    while idx < len(data):
        control = data[idx]
        idx += 1

        if control < 0:
            # Powtórzenie: -(N-1) -> N = -control + 1
            n = -control + 1
            value = data[idx]
            decoded.extend([value] * n)
            idx += 1
        else:
            # Literał: (N-1) -> N = control + 1
            n = control + 1
            decoded.extend(data[idx: idx + n])
            idx += n

    return np.array(decoded).reshape(shape)

a = np.dstack([np.eye(7),np.eye(7),np.eye(7)])
encoded = rle_encoder(a)
print("Encoded:", encoded)
decoded = rle_decoder(encoded)
print("Decoded:\n", decoded)
