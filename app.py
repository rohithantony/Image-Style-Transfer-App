import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os

# Load model
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Style and output images
PRESET_STYLES = [f'styles/style{i}.jpg' for i in range(1, 6)]
PREVIEW_OUTPUTS = [f'stylized/stylized{i}.jpg' for i in range(1, 6)]

class StyleTransferApp:
    def __init__(self, root):
        self.root = root
        root.title("Style Transfer")
        root.geometry("1400x850")
        root.configure(bg="black")

        self.content_image = None
        self.style_image = None

        # Sidebar toggle holder
        self.sidebar_holder = tb.Frame(root)
        self.sidebar_holder.pack(side="left", fill="y")

        # Sidebar pane
        self.left_frame = tb.Frame(self.sidebar_holder, width=300)
        self.left_frame.pack(side="top", fill="y")

        self.toggle_btn = tb.Button(self.sidebar_holder, text="Hide Pane", command=self.toggle_pane, bootstyle="warning-outline")
        self.toggle_btn.pack(pady=5)

        self.preset_label = tb.Label(self.left_frame, text="Preset Styles", font=("Segoe UI Variable Display", 16, "bold"), bootstyle="info")
        self.preset_label.pack(pady=(10, 2))
        self.style_carousel = tb.Label(self.left_frame)
        self.style_carousel.pack(pady=10)

        self.preview_label = tb.Label(self.left_frame, text="Stylized Images", font=("Segoe UI Variable Display", 16, "bold"), bootstyle="success")
        self.preview_label.pack(pady=(30, 2))
        self.preview_carousel = tb.Label(self.left_frame)
        self.preview_carousel.pack(pady=10)

        # Main content frame
        self.main_frame = tb.Frame(root)
        self.main_frame.pack(expand=True, fill="both")

        self.title_label = tb.Label(self.main_frame, text="Style Transfer", font=("Segoe UI Variable Display", 48, "bold"), bootstyle="info")
        self.title_label.pack(pady=(30, 10))

        self.desc = tb.Label(
            self.main_frame,
            text="Upload your image, choose a style, and transform it into art.",
            font=("Segoe UI Variable Display", 18), bootstyle="secondary"
        )
        self.desc.pack(pady=(0, 30))

        self.upload_content = tb.Button(self.main_frame, text="1. Upload Content Image", bootstyle="primary-outline", command=self.load_content, width=40)
        self.upload_content.pack(pady=15)

        self.upload_style = tb.Button(self.main_frame, text="2. Upload Style Image", bootstyle="info-outline", command=self.load_style, width=40)
        self.upload_style.pack(pady=15)

        self.stylize_button = tb.Button(self.main_frame, text="3. Stylize", bootstyle="success", command=self.stylize, state="disabled", width=40)
        self.stylize_button.pack(pady=15)

        # Image previews
        preview_frame = tb.Frame(self.main_frame)
        preview_frame.pack(pady=20)

        self.content_label = tb.Label(preview_frame)
        self.content_label.grid(row=0, column=0, padx=20)

        self.style_label = tb.Label(preview_frame)
        self.style_label.grid(row=0, column=1, padx=20)

        self.result_label = tb.Label(preview_frame)
        self.result_label.grid(row=0, column=2, padx=20)

        # Footer
        self.footer = tb.Label(self.main_frame, text="By Rohith Antony, 2025", font=("Segoe UI Variable Display", 12, "italic"), bootstyle="secondary")
        self.footer.pack(side="bottom", pady=15)

        self.style_index = 0
        self.preview_index = 0
        self.rotate_styles()
        self.rotate_previews()

    def toggle_pane(self):
        if self.left_frame.winfo_ismapped():
            self.left_frame.pack_forget()
            self.toggle_btn.config(text="Show Pane")
        else:
            self.left_frame.pack(side="top", fill="y")
            self.toggle_btn.config(text="Hide Pane")

    def load_image(self, path, label, max_size=350):
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_size, max_size))
        tk_img = ImageTk.PhotoImage(img)
        label.configure(image=tk_img)
        label.image = tk_img
        return img

    def load_content(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if path:
            self.content_image = self.load_image(path, self.content_label)
            self.check_ready()

    def load_style(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if path:
            self.style_image = self.load_image(path, self.style_label)
            self.check_ready()

    def prepare_tensor(self, pil_img):
        img = np.array(pil_img).astype(np.float32) / 255.0
        return tf.convert_to_tensor(img[None, ...])

    def tensor_to_pil(self, tensor):
        img = np.array(tensor[0] * 255, dtype=np.uint8)
        return Image.fromarray(img)

    def stylize(self):
        self.stylize_button.config(text="Processing...", state="disabled")
        self.root.update_idletasks()

        content_tensor = self.prepare_tensor(self.content_image)
        style_tensor = self.prepare_tensor(self.style_image)
        output = hub_model(content_tensor, style_tensor)[0]
        result = self.tensor_to_pil(output)
        result.thumbnail((350, 350))
        result_tk = ImageTk.PhotoImage(result)
        self.result_label.config(image=result_tk)
        self.result_label.image = result_tk

        self.stylize_button.config(text="3. Stylize", state="normal")

    def check_ready(self):
        if self.content_image and self.style_image:
            self.stylize_button.config(state="normal")

    def rotate_styles(self):
        path = PRESET_STYLES[self.style_index % len(PRESET_STYLES)]
        img = Image.open(path).resize((240, 240))
        tk_img = ImageTk.PhotoImage(img)
        self.style_carousel.config(image=tk_img)
        self.style_carousel.image = tk_img
        self.style_carousel.bind("<Button-1>", lambda e, p=path: self.load_style_from_path(p))
        self.style_index += 1
        self.root.after(5000, self.rotate_styles)

    def rotate_previews(self):
        path = PREVIEW_OUTPUTS[self.preview_index % len(PREVIEW_OUTPUTS)]
        img = Image.open(path).resize((240, 240))
        tk_img = ImageTk.PhotoImage(img)
        self.preview_carousel.config(image=tk_img)
        self.preview_carousel.image = tk_img
        self.preview_index += 1
        self.root.after(6000, self.rotate_previews)

    def load_style_from_path(self, path):
        self.style_image = self.load_image(path, self.style_label)
        self.check_ready()

if __name__ == "__main__":
    app = tb.Window(themename="darkly")
    StyleTransferApp(app)
    app.mainloop()
