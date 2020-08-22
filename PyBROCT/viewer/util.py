def quantized_ceil(v, q):
    if v % q == 0:
        return v
    else:
        return int(v // q + 1) * q

def quantized_floor(v, q):
    return int(v // q) * q

def quantized_remainder(v, d):
    return (int(v // d), int(v % d))

def sign(x):
    return -1 if x < 0 else 1
