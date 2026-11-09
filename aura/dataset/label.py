"""Tkinter helper for manually captioning AffectNet faces."""

from __future__ import annotations

import os
import tkinter as tk

from PIL import Image, ImageTk

from aura.config import CONFIG
from aura.dataset.dataset_provider import DatasetProvider
from aura.utils import hash_image


class ImageLabeler:
    """Simple UI for writing captions against AffectNet samples."""

    def __init__(self, root: tk.Tk, provider: DatasetProvider) -> None:
        self.root = root
        self.root.bind("<Return>", lambda _event: self.submit_label())
        self.provider = provider
        self.image_iter = iter(provider.dataset)
        self.current_hash: str | None = None
        self.current_image = None

        self.panel = tk.Label(root)
        self.panel.pack()

        self.emotion_label = tk.Label(root, text="")
        self.emotion_label.pack()

        self.label_entry = tk.Entry(root)
        self.label_entry.pack()

        self.submit_button = tk.Button(
            root, text="Submit", command=self.submit_label
        )
        self.submit_button.pack(pady=(0, 20))
        self.show_next_image()

    def show_next_image(self) -> None:
        """Advance to the next unlabeled face."""
        try:
            orig_img, _label, emotion = next(self.image_iter)
            self.current_image = orig_img
            self.current_hash = hash_image(orig_img)
            self.emotion_label.config(
                text=(
                    f"Hash: '{self.current_hash}', "
                    f"Emotion: {emotion.capitalize()}"
                )
            )
            img = Image.fromarray(orig_img[:, :, ::-1]).resize((400, 400))
            img_tk = ImageTk.PhotoImage(img)
            self.panel.config(image=img_tk)
            self.panel.image = img_tk
        except StopIteration:
            self.panel.config(text="No more images.")
            self.panel.image = None
            self.current_hash = None
            self.current_image = None

    def submit_label(self) -> None:
        """Persist the typed caption and move on."""
        label = self.label_entry.get().strip()
        if self.current_hash and label:
            self._save_caption(self.current_hash, label)
            print(f"Saved caption for {self.current_hash}: {label}")
        self.label_entry.delete(0, tk.END)
        self.show_next_image()

    @staticmethod
    def _save_caption(image_hash: str, caption: str) -> None:
        dataset_dir = os.path.join(CONFIG.aura_storage, "face_captions")
        img_folder = os.path.join(dataset_dir, image_hash)
        os.makedirs(img_folder, exist_ok=True)
        caption_path = os.path.join(img_folder, "caption.txt")
        with open(caption_path, "w", encoding="utf-8") as handle:
            handle.write(caption)


if __name__ == "__main__":
    provider = DatasetProvider()
    root = tk.Tk()
    root.title("Image Labeling")
    ImageLabeler(root, provider)
    root.mainloop()
