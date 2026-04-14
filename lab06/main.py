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
    pass

print(_search_for_non_repetitions(np.array([1,2,1,1,1,1,1,1,1,1,1,1,1,1,1])))
