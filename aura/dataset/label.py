import os
import hashlib
import base64

import tkinter as tk
import numpy as np
from PIL import Image, ImageTk

from aura.dataset import DatasetProvider


class ImageLabeler:
    def __init__(self, root, provider):
        self.root = root
        self.root.bind('<Return>', lambda _ : self.submit_label())
        self.provider = provider
        self.image_iter = iter(provider.dataset)

        self.panel = tk.Label(root)
        self.panel.pack()

        self.emotion_label = tk.Label(root, text = "")
        self.emotion_label.pack()

        self.label_entry = tk.Entry(root)
        self.label_entry.pack()

        self.submit_button = tk.Button(root, text="Submit", command=self.submit_label)
        self.submit_button.pack(pady = (0, 20))

        self.show_next_image()

    def show_next_image(self):
        try:
            orig_img, _, emotion = next(self.image_iter)
            img_hash = ImageLabeler._hash_image(orig_img)
            self.emotion_label.config(text = f"Hash: '{img_hash}', Emotion: {emotion.capitalize()}")
            img = Image.fromarray(orig_img[:, :, ::-1]) # Flipping channels (the last dimension) from 'BGR' -> 'RGB'
            img = img.resize((400, 400))
            img_tk = ImageTk.PhotoImage(img)
            self.panel.config(image=img_tk)
            self.panel.image = img_tk
        except StopIteration:
            self.panel.config(text="No more images.")
            self.panel.image = None

    def submit_label(self):
        label = self.label_entry.get()
        print(f"Label: {label}")
        self.label_entry.delete(0, tk.END)
        self.show_next_image()
    
    @staticmethod
    def _hash_image(img: np.ndarray, length: int = 10) -> str:
        arr_bytes = img.tobytes()
        sha1_hash = hashlib.sha1(arr_bytes).digest()  # SHA-1 produces a 20-byte hash
        base32_encoded = base64.b32encode(sha1_hash).decode('utf-8')
        return base32_encoded[:length]
    
    @staticmethod
    def _save_image(hash: str, caption: str):
        dataset_dir = os.path.join(os.getenv("STORAGE_PATH"), "aura_storage", "face_captions")
        os.makedirs(dataset_dir, exist_ok = True)
        img_folder = os.path.join(dataset_dir, hash)
        try:
            os.makedirs(img_folder)
        except:
            with open(os.path.join(img_folder, "caption.txt"), 'r', encoding='utf-8') as file:
                content = file.read()
            print(f"Image has already been labeled with '{content}'.")


if __name__ == "__main__":
    provider = DatasetProvider()
    root = tk.Tk()
    root.title("Image Labeling")
    app = ImageLabeler(root, provider)
    root.mainloop()