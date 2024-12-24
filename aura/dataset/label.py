import tkinter as tk
from PIL import Image, ImageTk

from aura.dataset import DatasetProvider


class ImageLabeler:
    def __init__(self, root, provider):
        self.root = root
        self.provider = provider
        self.image_iter = iter(provider.train)
        self.panel = tk.Label(root)
        self.panel.pack()
        self.label_entry = tk.Entry(root)
        self.label_entry.pack()
        self.submit_button = tk.Button(root, text="Submit", command=self.submit_label)
        self.submit_button.pack()
        self.show_next_image()

    def show_next_image(self):
        try:
            img_path = next(self.image_iter)
            img = Image.open(img_path)
            img = img.resize((250, 250), Image.ANTIALIAS)
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

if __name__ == "__main__":
    provider = DatasetProvider()
    root = tk.Tk()
    root.title("Image Labeling")
    app = ImageLabeler(root, provider)
    root.mainloop()