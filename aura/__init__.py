import os

import aura

from dotenv import load_dotenv


load_dotenv()
if not os.environ.get("STORAGE_PATH"):
    os.makedirs("output", exist_ok = True)
    os.environ["STORAGE_PATH"] = "./output"

save_dir = os.path.join(os.environ.get("STORAGE_PATH"), "aura_storage")
os.makedirs(save_dir, exist_ok = True)

__doc__ = aura.__doc__
if hasattr(aura, "__all__"):
    __all__ = aura.__all__
