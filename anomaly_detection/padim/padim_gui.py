import os
import sys
import glob
import shutil
import subprocess
import platform
import threading
import gc  # For manual garbage collection

from logging import getLogger

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from ttkthemes import ThemedTk

# Append paths for utility modules
sys.path.append('../../util')
import ailia
import log_init
from padim_utils import *
from model_utils import check_and_download_models
import webcamera_utils
from arg_utils import get_base_parser, update_parser

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/padim/'
logger = getLogger(__name__)


class CameraSelectionDialog(simpledialog.Dialog):
    """
    Custom dialog for selecting an available camera.
    """
    def __init__(self, parent: tk.Tk, cameras: list[str], title: str = "Select Camera") -> None:
        self.cameras = cameras
        self.selected: str | None = None
        super().__init__(parent, title=title)
    
    def body(self, master: tk.Frame) -> ttk.Combobox:
        ttk.Label(master, text="Select one of the available cameras:").grid(row=0, column=0, padx=5, pady=5)
        self.combo = ttk.Combobox(master, values=self.cameras, state="readonly")
        self.combo.grid(row=1, column=0, padx=5, pady=5)
        if self.cameras:
            self.combo.current(0)
        return self.combo

    def apply(self) -> None:
        self.selected = self.combo.get()


def create_limited_combobox(parent: tk.Widget, values: list[str], state: str, label_text: str, default_index: int = 0) -> ttk.Combobox:
    """
    Create a combobox with a width limited by the provided label text.
    """
    width = max(len(label_text) - 5, 10)
    cb = ttk.Combobox(parent, values=values, state=state, width=width)
    cb.current(default_index)
    return cb


class PaDiMApp(ThemedTk):
    """
    PaDiM GUI application built with ThemedTk.
    """
    def __init__(self) -> None:
        super().__init__(theme="arc")
        self.title("PaDiM GUI")
        self._init_window_geometry()
        self._init_state_variables()
        self._setup_menu()
        self._setup_status_bar()
        self._setup_gui()
        self.set_default_optimization_device()
        self.load_initial_lists()
        self.after(100, self.adjust_slider_length)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.bind("<KeyPress-q>", lambda event: self.stop_testing())
        self.frame_count = 0  # For periodic garbage collection

    def _init_window_geometry(self) -> None:
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        width = int(screen_width * 0.8)
        height = int(screen_height * 0.8)
        self.geometry(f"{width}x{height}")

    def _init_state_variables(self) -> None:
        self.train_folder: str = "train"
        self.test_folder: str | None = None
        self.test_type: str = "folder"
        self.test_roi: str | None = None
        self.score_cache: dict = {}
        self.stop_video: bool = False

    def on_closing(self) -> None:
        """Signal to stop background threads and close the application."""
        self.stop_video = True
        self._release_video_resources()
        self.destroy()

    def stop_testing(self) -> None:
        """Stop ongoing testing (video or camera) and re-enable buttons."""
        self.stop_video = True
        self.update_status("Testing stopped.")
        self.enable_buttons()

    def set_default_optimization_device(self) -> None:
        if torch.cuda.is_available():
            self.device_combobox.current(1)  # "cuda:0"
        else:
            self.device_combobox.current(0)  # "cpu"

    def adjust_slider_length(self) -> None:
        right_width = self.right_frame.winfo_width()
        new_length = max(right_width - 10, 50)
        self.slider.config(length=new_length)

    def _setup_menu(self) -> None:
        menu_bar = tk.Menu(self)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open Train Folder", command=self.train_folder_dialog)
        file_menu.add_command(label="Open Train Video", command=self.train_file_dialog)
        file_menu.add_command(label="Open Train Camera", command=self.train_camera_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Open Test Folder", command=self.test_folder_dialog)
        file_menu.add_command(label="Open Test Video", command=self.test_file_dialog)
        file_menu.add_command(label="Open Test Camera", command=self.test_camera_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Save Results", command=self.save_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        menu_bar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menu_bar)

    def _setup_status_bar(self) -> None:
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def update_status(self, message: str) -> None:
        self.status_var.set(message)
        self.update_idletasks()

    def _setup_gui(self) -> None:
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(0, weight=0)
        main_frame.columnconfigure(1, weight=2)
        main_frame.columnconfigure(2, weight=0)
        main_frame.rowconfigure(0, weight=1)

        self._create_left_frame(main_frame)
        self._create_center_frame(main_frame)
        self._create_right_frame(main_frame)

    def _create_left_frame(self, parent: ttk.Frame) -> None:
        self.left_frame = ttk.Frame(parent)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        ttk.Label(self.left_frame, text="Train Images").pack(anchor="w")
        # Train listbox with scrollbar.
        train_scroll_frame = ttk.Frame(self.left_frame)
        train_scroll_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        self.train_listbox = tk.Listbox(train_scroll_frame, height=10, exportselection=False)
        train_scroll = ttk.Scrollbar(train_scroll_frame, orient=tk.VERTICAL, command=self.train_listbox.yview)
        self.train_listbox.config(yscrollcommand=train_scroll.set)
        self.train_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        train_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.train_listbox.bind("<<ListboxSelect>>", self.on_train_select)
        self.train_listbox.bind("<Double-Button-1>", self.on_train_double_click)
        # Train control buttons.
        train_buttons = ttk.Frame(self.left_frame)
        train_buttons.pack(fill=tk.X, pady=5)
        ttk.Button(train_buttons, text="Open Folder", command=self.train_folder_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(train_buttons, text="Open Video", command=self.train_file_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(train_buttons, text="Open Camera", command=self.train_camera_dialog).pack(side=tk.LEFT, padx=2)

        ttk.Label(self.left_frame, text="Test Images/Video").pack(anchor="w", pady=(10, 0))
        # Test listbox with scrollbar.
        test_scroll_frame = ttk.Frame(self.left_frame)
        test_scroll_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        self.test_listbox = tk.Listbox(test_scroll_frame, height=10, exportselection=False)
        test_scroll = ttk.Scrollbar(test_scroll_frame, orient=tk.VERTICAL, command=self.test_listbox.yview)
        self.test_listbox.config(yscrollcommand=test_scroll.set)
        self.test_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        test_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.test_listbox.bind("<<ListboxSelect>>", self.on_test_select)
        self.test_listbox.bind("<Double-Button-1>", self.on_test_double_click)
        # Test control buttons.
        test_buttons = ttk.Frame(self.left_frame)
        test_buttons.pack(fill=tk.X, pady=5)
        ttk.Button(test_buttons, text="Open Folder", command=self.test_folder_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(test_buttons, text="Open Video", command=self.test_file_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(test_buttons, text="Open Camera", command=self.test_camera_dialog).pack(side=tk.LEFT, padx=2)

        # Fix left frame width.
        self.update_idletasks()
        btn1 = ttk.Button(self.left_frame, text="Open Folder")
        btn2 = ttk.Button(self.left_frame, text="Open Video")
        btn3 = ttk.Button(self.left_frame, text="Open Camera")
        left_width = btn1.winfo_reqwidth() + btn2.winfo_reqwidth() + btn3.winfo_reqwidth() + 10
        self.left_frame.config(width=left_width)
        self.left_frame.grid_propagate(False)
        parent.columnconfigure(0, minsize=left_width)

    def _create_center_frame(self, parent: ttk.Frame) -> None:
        self.center_frame = ttk.Frame(parent)
        self.center_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.center_frame.rowconfigure(0, weight=1)
        self.center_frame.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(self.center_frame, bg="black")
        self.canvas.grid(row=0, column=0, sticky="nsew")

    def _create_right_frame(self, parent: ttk.Frame) -> None:
        self.right_frame = ttk.Frame(parent)
        self.right_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        ttk.Label(self.right_frame, text="Result Images").pack(anchor="w")
        # Result listbox with scrollbar.
        result_scroll_frame = ttk.Frame(self.right_frame)
        result_scroll_frame.pack(fill=tk.BOTH, expand=False, pady=2)
        self.result_listbox = tk.Listbox(result_scroll_frame, height=10, exportselection=False)
        result_scroll = ttk.Scrollbar(result_scroll_frame, orient=tk.VERTICAL, command=self.result_listbox.yview)
        self.result_listbox.config(yscrollcommand=result_scroll.set)
        self.result_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        result_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_listbox.bind("<<ListboxSelect>>", self.on_result_select)
        self.result_listbox.bind("<Double-Button-1>", self.on_result_double_click)
        # Action buttons.
        actions_frame = ttk.Frame(self.right_frame)
        actions_frame.pack(fill=tk.X, pady=5)
        self.train_button = ttk.Button(actions_frame, text="Train", command=self.threaded_train)
        self.train_button.pack(side=tk.LEFT, padx=2)
        self.test_button = ttk.Button(actions_frame, text="Test", command=self.threaded_test)
        self.test_button.pack(side=tk.LEFT, padx=2)
        self.stop_test_button = ttk.Button(actions_frame, text="Stop Testing", command=self.stop_testing)
        self.stop_test_button.pack(side=tk.LEFT, padx=2)
        self.save_button = ttk.Button(actions_frame, text="Save Images", command=self.save_dialog)
        self.save_button.pack(side=tk.LEFT, padx=2)
        # ROI preview.
        roi_frame_right = ttk.LabelFrame(self.right_frame, text="ROI Preview")
        roi_frame_right.pack(fill=tk.X, pady=5, padx=5)
        self.canvas_roi = tk.Canvas(roi_frame_right, bg="black", width=140, height=140)
        self.canvas_roi.pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(roi_frame_right, text="Select ROI", command=self.test_roi_dialog).pack(side=tk.LEFT, padx=5, pady=5)
        # Settings frame.
        settings_frame = ttk.LabelFrame(self.right_frame, text="Settings")
        settings_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)
        pad_x, pad_y = 1, 1
        self.keep_aspect = tk.BooleanVar(value=True)
        self.center_crop = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Keep Aspect", variable=self.keep_aspect).grid(row=0, column=0, sticky="w", padx=pad_x, pady=pad_y)
        ttk.Checkbutton(settings_frame, text="Center Crop", variable=self.center_crop).grid(row=0, column=1, sticky="w", padx=pad_x, pady=pad_y)
        # Feature extractor model selection.
        model_label = "Feature Extractor Model"
        ttk.Label(settings_frame, text=model_label).grid(row=1, column=0, columnspan=2, sticky="w", padx=pad_x, pady=(5, pad_y))
        self.model_list = ["resnet18 (224)", "resnet18 (448)", "wide_resnet50_2 (224)"]
        self.model_combobox = create_limited_combobox(settings_frame, self.model_list, "readonly", model_label, default_index=1)
        self.model_combobox.grid(row=2, column=0, columnspan=2, sticky="w", padx=pad_x, pady=pad_y)
        # Optimization and file format settings.
        self.enable_optimization = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Enable Optimization", variable=self.enable_optimization).grid(row=3, column=0, columnspan=2, sticky="w", padx=pad_x, pady=pad_y)
        file_format_label = "Trained File Format"
        ttk.Label(settings_frame, text=file_format_label).grid(row=4, column=0, sticky="w", padx=pad_x, pady=(5, pad_y))
        self.file_format_list = ["pkl", "pt", "npy"]
        self.file_format_combobox = create_limited_combobox(settings_frame, self.file_format_list, "readonly", file_format_label, default_index=1)
        self.file_format_combobox.grid(row=5, column=0, sticky="w", padx=pad_x, pady=pad_y)
        device_label = "Optimization Device"
        ttk.Label(settings_frame, text=device_label).grid(row=4, column=1, sticky="w", padx=pad_x, pady=(5, pad_y))
        self.device_list = ["cpu", "cuda:0", "mps"]
        self.device_combobox = create_limited_combobox(settings_frame, self.device_list, "readonly", device_label, default_index=0)
        self.device_combobox.grid(row=5, column=1, sticky="w", padx=pad_x, pady=pad_y)
        # Threshold slider.
        ttk.Label(settings_frame, text="Threshold", anchor="center").grid(row=6, column=0, columnspan=2, sticky="ew", padx=pad_x, pady=(5, pad_y))
        self.threshold = tk.IntVar(value=50)
        self.slider = tk.Scale(settings_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                               variable=self.threshold, showvalue=True, length=300, sliderlength=30)
        self.slider.grid(row=7, column=0, columnspan=2, padx=pad_x, pady=pad_y)

        # Limit right frame width.
        self.update_idletasks()
        rbtn1 = ttk.Button(self.right_frame, text="Train")
        rbtn2 = ttk.Button(self.right_frame, text="Test")
        rbtn3 = ttk.Button(self.right_frame, text="Save Images")
        right_width = rbtn1.winfo_reqwidth() + rbtn2.winfo_reqwidth() + rbtn3.winfo_reqwidth() + 10
        self.right_frame.config(width=right_width)
        self.right_frame.grid_propagate(False)
        parent.columnconfigure(2, minsize=right_width)
        settings_frame.config(width=right_width)
        settings_frame.grid_propagate(False)

    def load_initial_lists(self) -> None:
        self.train_list = self.get_training_file_list()
        self.test_list = self.get_test_file_list()
        self.result_list = []
        self.train_listbox.delete(0, tk.END)
        for f in self.train_list:
            self.train_listbox.insert(tk.END, os.path.basename(f))
        if self.train_list:
            self.train_listbox.selection_set(0)
            self.on_train_select(None)
        self.test_listbox.delete(0, tk.END)
        for f in self.test_list:
            self.test_listbox.insert(tk.END, os.path.basename(f))

    def on_train_select(self, event: tk.Event | None) -> None:
        selection = self.train_listbox.curselection()
        if selection:
            index = selection[0]
            self.load_detail(self.train_list[index])

    def on_test_select(self, event: tk.Event | None) -> None:
        selection = self.test_listbox.curselection()
        if selection:
            index = selection[0]
            self.load_detail(self.test_list[index])

    def on_result_select(self, event: tk.Event | None) -> None:
        selection = self.result_listbox.curselection()
        if selection:
            index = selection[0]
            self.load_detail(self.result_list[index])

    def on_train_double_click(self, event: tk.Event) -> None:
        selection = self.train_listbox.curselection()
        if selection:
            self.open_file_by_os(self.train_list[selection[0]])

    def on_test_double_click(self, event: tk.Event) -> None:
        selection = self.test_listbox.curselection()
        if selection:
            self.open_file_by_os(self.test_list[selection[0]])

    def on_result_double_click(self, event: tk.Event) -> None:
        selection = self.result_listbox.curselection()
        if selection:
            self.open_file_by_os(self.result_list[selection[0]])

    def open_file_by_os(self, filepath: str) -> None:
        try:
            if platform.system() == 'Darwin':
                subprocess.call(('open', filepath))
            elif platform.system() == 'Windows':
                os.startfile(filepath)
            else:
                subprocess.call(('xdg-open', filepath))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def create_photo_image(self, path: str, canvas: tk.Canvas | None = None) -> ImageTk.PhotoImage:
        if canvas is None:
            canvas = self.canvas
        canvas.update_idletasks()
        w = canvas.winfo_width() or 480
        h = canvas.winfo_height() or 160
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            cap = cv2.VideoCapture(path)
            ret, image_bgr = cap.read()
            cap.release()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_pil.thumbnail((w, h), Image.LANCZOS)
        return ImageTk.PhotoImage(image_pil)

    def load_detail(self, image_path: str) -> None:
        if os.path.exists(image_path):
            try:
                img_tk = self.create_photo_image(image_path, canvas=self.canvas)
                self.canvas.delete("all")
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                self.canvas.create_image(canvas_width/2, canvas_height/2, image=img_tk, anchor="center")
                self.canvas.image = img_tk
            except Exception as e:
                self.update_status(f"Error loading image: {e}")

    def train_file_dialog(self) -> None:
        file_path = filedialog.askopenfilename(filetypes=[("Image/Video Files", "*")])
        if file_path:
            self.train_folder = file_path
            self.train_list = [file_path]
            self.train_listbox.delete(0, tk.END)
            self.train_listbox.insert(tk.END, os.path.basename(file_path))
            self.load_detail(file_path)

    def train_folder_dialog(self) -> None:
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.train_folder = folder_path
            self.train_list = self.get_training_file_list()
            self.train_listbox.delete(0, tk.END)
            for f in self.train_list:
                self.train_listbox.insert(tk.END, os.path.basename(f))
            if self.train_list:
                self.load_detail(self.train_list[0])

    def train_camera_dialog(self) -> None:
        cameras = self.get_camera_list()
        if not cameras:
            messagebox.showerror("Error", "No camera detected.")
            return
        dialog = CameraSelectionDialog(self, cameras, title="Select Train Camera")
        if dialog.selected is None:
            return
        index = dialog.selected.split(":")[1]
        cap = cv2.VideoCapture(int(index))
        if not cap.isOpened():
            messagebox.showerror("Error", f"Camera with index {index} not detected.")
            cap.release()
            return
        cap.release()
        self.train_folder = "camera"
        self.train_list = [f"camera:{index}"]
        self.train_listbox.delete(0, tk.END)
        self.train_listbox.insert(tk.END, f"camera:{index}")
        self.load_detail(self.train_list[0])

    def test_file_dialog(self) -> None:
        file_path = filedialog.askopenfilename(filetypes=[("Image/Video Files", "*")])
        if file_path:
            self.test_folder = file_path
            self.test_list = [file_path]
            self.test_listbox.delete(0, tk.END)
            self.test_listbox.insert(tk.END, os.path.basename(file_path))
            self.load_detail(file_path)
            self.test_type = "video"

    def test_folder_dialog(self) -> None:
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.test_folder = folder_path
            self.test_list = self.get_test_file_list()
            self.test_listbox.delete(0, tk.END)
            for f in self.test_list:
                self.test_listbox.insert(tk.END, os.path.basename(f))
            if self.test_list:
                self.load_detail(self.test_list[0])
            self.test_type = "folder"

    def test_camera_dialog(self) -> None:
        cameras = self.get_camera_list()
        if not cameras:
            messagebox.showerror("Error", "No camera detected.")
            return
        dialog = CameraSelectionDialog(self, cameras, title="Select Test Camera")
        if dialog.selected is None:
            return
        index = dialog.selected.split(":")[1]
        cap = cv2.VideoCapture(int(index))
        if not cap.isOpened():
            messagebox.showerror("Error", f"Camera with index {index} not detected.")
            cap.release()
            return
        cap.release()
        self.test_folder = "camera"
        self.test_list = [f"camera:{index}"]
        self.test_listbox.delete(0, tk.END)
        self.test_listbox.insert(tk.END, f"camera:{index}")
        self.load_detail(self.test_list[0])
        self.test_type = "video"

    def test_roi_dialog(self) -> None:
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*")])
        if file_path:
            self.test_roi = file_path
            img_tk = self.create_photo_image(file_path, canvas=self.canvas_roi)
            self.canvas_roi.delete("all")
            self.canvas_roi.create_image(0, 0, image=img_tk, anchor=tk.NW)
            self.canvas_roi.image = img_tk

    def save_dialog(self) -> None:
        folder_path = filedialog.askdirectory()
        if folder_path:
            try:
                shutil.copytree('result', folder_path, dirs_exist_ok=True)
                self.update_status(f"Results saved to {folder_path}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def get_camera_list(self) -> list[str]:
        index = 0
        cameras = []
        while True:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                cameras.append(f"camera:{index}")
            else:
                cap.release()
                break
            cap.release()
            index += 1
        return cameras

    def get_file_list(self, folder: str) -> list[str]:
        files = glob.glob(os.path.join(folder, "*.jpg"))
        files.extend(glob.glob(os.path.join(folder, "*.png")))
        files.extend(glob.glob(os.path.join(folder, "*.bmp")))
        return sorted(files)

    def get_training_file_list(self) -> list[str]:
        if self.train_folder and os.path.isdir(self.train_folder):
            return self.get_file_list(self.train_folder)
        return []

    def get_test_file_list(self) -> list[str]:
        if self.test_folder and os.path.isdir(self.test_folder):
            return self.get_file_list(self.test_folder)
        return ["bottle_000.png"]

    def threaded_train(self) -> None:
        threading.Thread(target=self.train_button_clicked, daemon=True).start()

    def threaded_test(self) -> None:
        threading.Thread(target=self.test_button_clicked, daemon=True).start()

    def train_button_clicked(self) -> None:
        self.update_status("Training started...")
        self.disable_buttons()
        try:
            selected_model = self.model_combobox.get().split()[0]
            self.weight_path, self.model_path, self.params = get_params(selected_model)
            check_and_download_models(self.weight_path, self.model_path, REMOTE_PATH)
            self.net = ailia.Net(self.model_path, self.weight_path, env_id=self.args.env_id)
            batch_size = 4
            train_dir = self.train_folder if self.train_folder != "camera" else self.train_list[0].split(":")[1]
            aug, aug_num, seed = False, 0, 1024
            train_outputs = training_optimized(
                self.net, self.params, self.get_image_resize(), self.get_image_crop_size(),
                self.keep_aspect.get(), batch_size, train_dir, aug, aug_num, seed, logger
            )
            self.trained_model_save(train_outputs)
            self.update_status("Training completed.")
        except Exception as e:
            messagebox.showerror("Training Error", str(e))
            self.update_status("Training failed.")
        finally:
            self.enable_buttons()
            gc.collect()  # Trigger garbage collection after training

    def trained_model_save(self, train_outputs: list) -> None:
        file_format = self.file_format_combobox.get()
        if not self.enable_optimization.get():
            if file_format == "pkl":
                self.train_feat_file = "train.pkl"
                logger.info(f"Saving train set feature to: {self.train_feat_file}")
                with open(self.train_feat_file, 'wb') as f:
                    pickle.dump(train_outputs, f)
            elif file_format == "npy":
                for i, output in enumerate(train_outputs):
                    self.train_feat_file = f"train_output_{i}.npy"
                    np.save(self.train_feat_file, output)
            elif file_format == "pt":
                self.train_feat_file = "train.pt"
                torch.save(train_outputs, self.train_feat_file)
        else:
            if file_format == "npy":
                for i, output in enumerate(train_outputs):
                    self.train_feat_file = f"train_output_{i}.npy"
                    np.save(self.train_feat_file, output)
            train_outputs = [
                torch.from_numpy(train_outputs[0]).float().to(self.device),
                train_outputs[1],
                torch.from_numpy(train_outputs[2]).float().to(self.device),
                train_outputs[3]
            ]
            if file_format == "pkl":
                self.train_feat_file = "trainOptimized.pkl"
                with open(self.train_feat_file, 'wb') as f:
                    pickle.dump(train_outputs, f)
            elif file_format == "pt":
                self.train_feat_file = "trainOptimized.pt"
                torch.save(train_outputs, self.train_feat_file)
        self.train_outputs = train_outputs

    def test_button_clicked(self) -> None:
        self.update_status("Testing started...")
        self.disable_buttons()
        self.stop_video = False
        # Clear score cache before starting a new test to free old results.
        self.score_cache.clear()
        try:
            if not hasattr(self, "net"):
                file_format = self.file_format_combobox.get()
                expected_file = None
                if file_format == "pkl":
                    expected_file = "trainOptimized.pkl" if self.enable_optimization.get() else "train.pkl"
                elif file_format == "pt":
                    expected_file = "trainOptimized.pt" if self.enable_optimization.get() else "train.pt"
                if expected_file and os.path.exists(expected_file):
                    with open(expected_file, "rb") as f:
                        self.train_outputs = pickle.load(f) if file_format == "pkl" else torch.load(f)
                    selected_model = self.model_combobox.get().split()[0]
                    self.weight_path, self.model_path, self.params = get_params(selected_model)
                    self.net = ailia.Net(self.model_path, self.weight_path, env_id=self.args.env_id)
                else:
                    ret = messagebox.askyesno("Model Not Trained", "No trained model found. Do you want to train now?")
                    if ret:
                        self.train_button_clicked()
                    else:
                        self.enable_buttons()
                        return
            threshold = self.threshold.get() / 100.0
            if self.test_type == "folder":
                self.test_from_folder(threshold)
                self.update_status("Testing completed.")
                self.enable_buttons()
            else:
                self.test_from_video(threshold)
            self.update_status("Testing completed.")
        except Exception as e:
            messagebox.showerror("Testing Error", str(e))
            self.update_status("Testing failed.")
            self.enable_buttons()
        finally:
            gc.collect()  # Clean up memory after testing

    def test_from_folder(self, threshold: float) -> None:
        # Clear score cache at the start
        self.score_cache.clear()
        test_imgs = []
        roi_img = None
        if self.test_roi:
            roi_img = load_image(self.test_roi)
            roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGRA2RGB)
            roi_img = preprocess(roi_img, self.get_image_resize(),
                                 keep_aspect=self.keep_aspect.get(),
                                 crop_size=self.get_image_crop_size(), mask=True)
        score_map = []
        for image_path in self.test_list:
            if self.stop_video:
                self.update_status("Testing stopped.")
                self.enable_buttons()
                return
            logger.info(f"Processing: {image_path}")
            img = load_image(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            img = preprocess(img, self.get_image_resize(), keep_aspect=self.keep_aspect.get(),
                             crop_size=self.get_image_crop_size())
            test_imgs.append(img[0])
            if self.enable_optimization.get():
                dist_tmp = infer_optimized(
                    self.net, self.params, self.train_outputs,
                    img=img, crop_size=self.get_image_crop_size(),
                    device=self.device, logger=logger, weights_torch=self.weights_torch
                )
                self.score_cache[image_path] = dist_tmp
            else:
                dist_tmp = infer(
                    self.net, self.params, self.train_outputs,
                    img=img, crop_size=self.get_image_crop_size()
                )
                self.score_cache[image_path] = dist_tmp.copy()
            score_map.append(dist_tmp)
            # Explicitly delete temporary variables if needed.
            del img, dist_tmp
        os.makedirs("result", exist_ok=True)
        self.result_list = []
        if self.enable_optimization.get():
            scores = normalize_scores_torch(score_map, self.get_image_crop_size(), roi_img)
            scores = np.asarray([scores[i].cpu().numpy()[0] for i in range(len(scores))])
        else:
            scores = normalize_scores(score_map, self.get_image_crop_size(), roi_img)
        for i, test_img in enumerate(test_imgs):
            img_denorm = denormalization(test_img)
            heat_map, mask, vis_img = visualize(img_denorm, scores[i], threshold)
            frame = pack_visualize_gui(heat_map, mask, vis_img, scores, self.get_image_crop_size())
            output_path = os.path.join("result", os.path.basename(self.test_list[i]))
            cv2.imwrite(output_path, frame)
            self.result_list.append(output_path)
        self.result_listbox.delete(0, tk.END)
        for f in self.result_list:
            self.result_listbox.insert(tk.END, os.path.basename(f))
        if self.result_list:
            self.load_detail(self.result_list[0])
        self.enable_buttons()

    def test_from_video(self, threshold: float) -> None:
        self.update_status("Testing video started...")
        self.video_result_path = "result.mp4"
        # Clear score cache to free previous results.
        self.score_cache.clear()
        video_path = self.test_folder if self.test_folder != "camera" else self.test_list[0].split(":")[1]
        self.capture = webcamera_utils.get_capture(video_path)
        self.crop_size = self.get_image_crop_size()
        frame_height = int(self.crop_size * 1.8)
        frame_width = int(self.crop_size * 2.8)
        self.writer = webcamera_utils.get_writer(self.video_result_path, frame_height, frame_width)
        self.stop_video = False
        self.frame_count = 0  # Reset frame counter
        self.process_video_frame(threshold, frame_width, frame_height)

    def process_video_frame(self, threshold: float, frame_width: int, frame_height: int) -> None:
        if self.stop_video:
            self._release_video_resources()
            self.enable_buttons()
            return

        ret, frame = self.capture.read()
        if not ret:
            self._release_video_resources()
            self.result_list = [self.video_result_path]
            self.result_listbox.delete(0, tk.END)
            self.result_listbox.insert(tk.END, os.path.basename(self.video_result_path))
            self.load_detail(self.video_result_path)
            self.update_status("Testing completed.")
            self.enable_buttons()
            return
        
        # Process the frame.
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_preprocessed = preprocess(
            img,
            self.get_image_resize(),
            keep_aspect=self.keep_aspect.get(),
            crop_size=self.crop_size
        )
        
        if self.enable_optimization.get():
            dist_tmp = infer_optimized(
                self.net, self.params, self.train_outputs,
                img=img_preprocessed, crop_size=self.crop_size,
                device=self.device, logger=logger, weights_torch=self.weights_torch
            )
            norm_scores_tensor = normalize_scores_torch([dist_tmp], self.crop_size, None)
            score_value = norm_scores_tensor[0].cpu().numpy()[0]
        else:
            dist_tmp = infer(
                self.net, self.params, self.train_outputs,
                img=img_preprocessed, crop_size=self.crop_size
            )
            norm_scores = normalize_scores([dist_tmp], self.crop_size, None)
            score_value = norm_scores[0]
        
        denorm_img = denormalization(img_preprocessed[0])
        heat_map, mask, vis_img = visualize(denorm_img, score_value, threshold)
        frame_out = pack_visualize_gui(heat_map, mask, vis_img, np.asarray([score_value]), self.crop_size)
        frame_out = cv2.resize(frame_out, (frame_width, frame_height))
        
        frame_rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)
        image_pil = image_pil.resize((frame_width, frame_height), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(image_pil)
        
        self.canvas.delete("all")
        canvas_width = self.canvas.winfo_width() or frame_width
        canvas_height = self.canvas.winfo_height() or frame_height
        self.canvas.create_image(canvas_width/2, canvas_height/2, image=img_tk, anchor="center")
        self.canvas.image = img_tk
        
        if self.writer is not None:
            self.writer.write(frame_out)
        
        # Clean up temporary variables.
        del img, img_preprocessed, dist_tmp, frame_out, image_pil, frame_rgb

        # Increment frame counter and periodically call gc.collect().
        self.frame_count += 1
        if self.frame_count % 100 == 0:
            gc.collect()

        self.after(10, lambda: self.process_video_frame(threshold, frame_width, frame_height))
    
    def _release_video_resources(self) -> None:
        if hasattr(self, "capture") and self.capture is not None:
            self.capture.release()
            self.capture = None
        if hasattr(self, "writer") and self.writer is not None:
            self.writer.release()
            self.writer = None

    def get_image_crop_size(self) -> int:
        index = self.model_combobox.current()
        resolutions = [224, 448, 224]
        return resolutions[index] if index < len(resolutions) else 224

    def get_image_resize(self) -> int:
        crop_size = self.get_image_crop_size()
        return crop_size + (256 - 224) if self.center_crop.get() else crop_size

    def disable_buttons(self) -> None:
        self.train_button.config(state=tk.DISABLED)
        self.test_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)

    def enable_buttons(self) -> None:
        self.train_button.config(state=tk.NORMAL)
        self.test_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)

    @property
    def device(self) -> torch.device:
        return torch.device(self.device_combobox.get())

    @property
    def weights_torch(self) -> torch.Tensor:
        w = gaussian_kernel1d_torch(4, 0, int(4.0 * 4 + 0.5), self.device)
        return w.unsqueeze(0).unsqueeze(0).expand(1, 1, 33)

    @property
    def args(self):
        if not hasattr(self, '_args'):
            parser = get_base_parser('PaDiM GUI', None, None)
            self._args = update_parser(parser)
        return self._args


if __name__ == '__main__':
    app = PaDiMApp()
    app.mainloop()
