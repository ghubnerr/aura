import base64
import hashlib

import numpy as np

def hash_image(img: np.ndarray, length: int = 10) -> str:
        arr_bytes = img.tobytes()
        sha1_hash = hashlib.sha1(arr_bytes).digest()  # SHA-1 produces a 20-byte hash
        base32_encoded = base64.b32encode(sha1_hash).decode('utf-8')
        return base32_encoded[:length]