import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from typing import Tuple, Optional
from out_of_the_box.bounding_boxes import BoundingBox, VOCBox, YOLOBox, COCOBox


# Constants
SCREEN_SCALE = 0.8
MIN_BOX_SIZE = 5
HANDLE_SIZE = 6
PADDING = 5


class ROISelector(tk.Frame):
    def __init__(self, master: tk.Tk):
        """Initialize the ROI Selector application."""
        super().__init__(master)
        self.master = master
        self.master.state('zoomed')  # Start maximized

        self.image: Optional[Image.Image] = None
        self.photo_image: Optional[ImageTk.PhotoImage] = None
        self.bounding_box: Optional[BoundingBox] = None
        self.image_position: Tuple[int, int] = (0, 0)
        self.scale_factor: float = 1.0

        self._setup_ui()
        self._bind_events()

    def _setup_ui(self):
        """Set up the user interface."""
        self.image_frame = tk.Frame(self)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.entry_frame = tk.Frame(self)
        self.entry_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=PADDING, pady=PADDING)

        open_button = tk.Button(self.entry_frame, text="Open Image", command=self.open_image)
        open_button.pack(pady=PADDING)

        self.canvas = tk.Canvas(self.image_frame)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.voc_entries = self._create_format_entries("VOC", ["xmin", "ymin", "xmax", "ymax", "xmin_norm", "ymin_norm",
                                                               "xmax_norm", "ymax_norm"])
        self.coco_entries = self._create_format_entries("COCO",
                                                        ["x", "y", "width", "height", "x_norm", "y_norm", "width_norm",
                                                         "height_norm"])
        self.yolo_entries = self._create_format_entries("YOLO",
                                                        ["x_center", "y_center", "width", "height", "x_center_norm",
                                                         "y_center_norm", "width_norm", "height_norm"])

    def _bind_events(self):
        """Bind mouse events to the canvas."""
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)

    def _create_format_entries(self, format_name: str, field_names: list) -> dict:
        """Create input fields for a specific bounding box format."""
        frame = tk.LabelFrame(self.entry_frame, text=format_name)
        frame.pack(fill=tk.X, padx=PADDING, pady=PADDING)
        entries = {}
        for i, name in enumerate(field_names):
            row = tk.Frame(frame)
            row.pack(fill=tk.X)
            tk.Label(row, text=f"{name}:").pack(side=tk.LEFT)
            entry = tk.Entry(row)
            entry.pack(side=tk.RIGHT, expand=True, fill=tk.X)
            entry.bind("<Return>", lambda event, f=format_name, n=name: self.on_entry_change(event, f, n))
            entries[name] = entry
            if i == 3:  # Add a separator between absolute and normalized values
                tk.Frame(frame, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, padx=PADDING, pady=PADDING)
        return entries

    def open_image(self):
        """Open an image file and display it on the canvas."""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if file_path:
            try:
                self.image = Image.open(file_path)
                self.update_image()
            except IOError:
                tk.messagebox.showerror("Error", "Failed to open the image file.")

    def update_image(self):
        """Update the displayed image and adjust the window size."""
        if self.image is not None:
            screen_width = self.master.winfo_screenwidth()
            screen_height = self.master.winfo_screenheight()

            img_width, img_height = self.image.size

            self.scale_factor = min(1.0, (screen_width * SCREEN_SCALE) / img_width,
                                    (screen_height * SCREEN_SCALE) / img_height)

            new_size = (int(img_width * self.scale_factor), int(img_height * self.scale_factor))
            resized_image = self.image.copy()
            resized_image.thumbnail(new_size, Image.LANCZOS)

            self.photo_image = ImageTk.PhotoImage(resized_image)

            window_width = new_size[0] + self.entry_frame.winfo_reqwidth()
            window_height = max(new_size[1], self.entry_frame.winfo_reqheight())
            self.master.geometry(f"{window_width}x{window_height}")

            self.canvas.config(width=new_size[0], height=new_size[1])

            self.image_position = ((new_size[0] - self.photo_image.width()) // 2,
                                   (new_size[1] - self.photo_image.height()) // 2)

            self.canvas.create_image(self.image_position[0], self.image_position[1], anchor=tk.NW,
                                     image=self.photo_image)

            self.reset_bounding_box()

    def reset_bounding_box(self):
        """Reset the bounding box and clear the canvas."""
        self.bounding_box = None
        self.canvas.delete("box")
        self.canvas.delete("handle")

    def start_draw(self, event):
        """Start drawing or resizing the bounding box."""
        x, y = self.get_image_coordinates(event)
        if x is not None and y is not None:
            self.start_x, self.start_y = x, y
            self.resize_handle = self.get_resize_handle(event.x, event.y)
            if self.resize_handle and self.bounding_box:
                voc = self.bounding_box.to_voc()
                self.original_box = (voc.xmin, voc.ymin, voc.xmax, voc.ymax)

    def draw(self, event):
        """Continue drawing or resizing the bounding box."""
        if self.start_x is not None and self.start_y is not None:
            x, y = self.get_image_coordinates(event)
            if x is not None and y is not None:
                if self.resize_handle:
                    self.resize_box(x, y)
                else:
                    self.canvas.delete("box")
                    self.canvas.delete("handle")
                    self.canvas.create_rectangle(
                        self.start_x * self.scale_factor + self.image_position[0],
                        self.start_y * self.scale_factor + self.image_position[1],
                        x * self.scale_factor + self.image_position[0],
                        y * self.scale_factor + self.image_position[1],
                        outline="red", tags="box"
                    )

    def end_draw(self, event):
        """Finish drawing or resizing the bounding box."""
        x, y = self.get_image_coordinates(event)
        if x is not None and y is not None:
            if self.resize_handle:
                self.resize_box(x, y)
            elif self.start_x is not None and self.start_y is not None:
                x1, y1 = min(self.start_x, x), min(self.start_y, y)
                x2, y2 = max(self.start_x, x), max(self.start_y, y)

                # Ensure minimum size
                if x2 - x1 < MIN_BOX_SIZE:
                    x2 = x1 + MIN_BOX_SIZE
                if y2 - y1 < MIN_BOX_SIZE:
                    y2 = y1 + MIN_BOX_SIZE

                self.update_bounding_box(x1, y1, x2, y2)

        self.start_x = self.start_y = self.resize_handle = self.original_box = None

    def get_image_coordinates(self, event) -> Tuple[Optional[int], Optional[int]]:
        """Convert canvas coordinates to image coordinates."""
        if self.image is None:
            return None, None

        x = (event.x - self.image_position[0]) / self.scale_factor
        y = (event.y - self.image_position[1]) / self.scale_factor

        img_width, img_height = self.image.size

        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))

        return int(x), int(y)

    def update_bounding_box(self, x1: int, y1: int, x2: int, y2: int):
        """Update the bounding box with new coordinates."""
        img_width, img_height = self.image.size

        # Ensure the box stays within the image boundaries
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))

        voc_box = VOCBox(x1, y1, x2, y2, normalized=False)
        self.bounding_box = BoundingBox(voc_box, (img_height, img_width))
        self.update_entries()
        self.draw_bounding_box()

    def draw_bounding_box(self):
        """Draw the bounding box and resize handles on the canvas."""
        if self.bounding_box:
            self.canvas.delete("box")
            self.canvas.delete("handle")
            voc = self.bounding_box.to_voc()
            x1 = voc.xmin * self.scale_factor + self.image_position[0]
            y1 = voc.ymin * self.scale_factor + self.image_position[1]
            x2 = voc.xmax * self.scale_factor + self.image_position[0]
            y2 = voc.ymax * self.scale_factor + self.image_position[1]

            self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", tags="box")

            # Draw resize handles
            handles = [
                (x1, y1), (x2, y1), (x2, y2), (x1, y2),  # corners
                ((x1 + x2) / 2, y1), (x2, (y1 + y2) / 2),  # top and right middle
                ((x1 + x2) / 2, y2), (x1, (y1 + y2) / 2)  # bottom and left middle
            ]
            for hx, hy in handles:
                self.canvas.create_rectangle(
                    hx - HANDLE_SIZE, hy - HANDLE_SIZE,
                    hx + HANDLE_SIZE, hy + HANDLE_SIZE,
                    fill="blue", outline="white", tags="handle"
                )

    def get_resize_handle(self, x: int, y: int) -> Optional[str]:
        """Get the resize handle at the given coordinates."""
        if not self.bounding_box:
            return None

        voc = self.bounding_box.to_voc()
        x1 = voc.xmin * self.scale_factor + self.image_position[0]
        y1 = voc.ymin * self.scale_factor + self.image_position[1]
        x2 = voc.xmax * self.scale_factor + self.image_position[0]
        y2 = voc.ymax * self.scale_factor + self.image_position[1]

        handles = {
            "top_left": (x1, y1), "top_right": (x2, y1),
            "bottom_right": (x2, y2), "bottom_left": (x1, y2),
            "top": ((x1 + x2) / 2, y1), "right": (x2, (y1 + y2) / 2),
            "bottom": ((x1 + x2) / 2, y2), "left": (x1, (y1 + y2) / 2)
        }

        for handle, (hx, hy) in handles.items():
            if abs(x - hx) <= HANDLE_SIZE and abs(y - hy) <= HANDLE_SIZE:
                return handle

        return None

    def resize_box(self, x: int, y: int):
        """Resize the bounding box based on the selected handle."""
        if not self.bounding_box or not self.original_box:
            return

        x1, y1, x2, y2 = self.original_box

        if self.resize_handle == "top_left":
            x1, y1 = x, y
        elif self.resize_handle == "top_right":
            x2, y1 = x, y
        elif self.resize_handle == "bottom_right":
            x2, y2 = x, y
        elif self.resize_handle == "bottom_left":
            x1, y2 = x, y
        elif self.resize_handle == "top":
            y1 = y
        elif self.resize_handle == "right":
            x2 = x
        elif self.resize_handle == "bottom":
            y2 = y
        elif self.resize_handle == "left":
            x1 = x

        # Ensure minimum size
        if x2 - x1 < MIN_BOX_SIZE:
            if self.resize_handle in ["top_left", "bottom_left", "left"]:
                x1 = x2 - MIN_BOX_SIZE
            else:
                x2 = x1 + MIN_BOX_SIZE
        if y2 - y1 < MIN_BOX_SIZE:
            if self.resize_handle in ["top_left", "top_right", "top"]:
                y1 = y2 - MIN_BOX_SIZE
            else:
                y2 = y1 + MIN_BOX_SIZE

        self.update_bounding_box(x1, y1, x2, y2)

    def update_entries(self):
        """Update the entry fields with the current bounding box values."""
        if self.bounding_box:
            voc = self.bounding_box.to_voc()
            voc_norm = self.bounding_box.to_voc(normalized=True)
            coco = self.bounding_box.to_coco()
            coco_norm = self.bounding_box.to_coco(normalized=True)
            yolo = self.bounding_box.to_yolo(normalized=False)
            yolo_norm = self.bounding_box.to_yolo()

            self._update_entry_values(self.voc_entries,
                                      [voc.xmin, voc.ymin, voc.xmax, voc.ymax,
                                       voc_norm.xmin, voc_norm.ymin, voc_norm.xmax, voc_norm.ymax])

            self._update_entry_values(self.coco_entries,
                                      [coco.x, coco.y, coco.width, coco.height,
                                       coco_norm.x, coco_norm.y, coco_norm.width, coco_norm.height])

            self._update_entry_values(self.yolo_entries,
                                      [yolo.x_center, yolo.y_center, yolo.width, yolo.height,
                                       yolo_norm.x_center, yolo_norm.y_center, yolo_norm.width, yolo_norm.height])

    def _update_entry_values(self, entries: dict, values: list):
        """Update the values of the entry fields."""
        for entry, value in zip(entries.values(), values):
            entry.delete(0, tk.END)
            entry.insert(0, f"{value:.4f}")

    def on_entry_change(self, event, format_name: str, field_name: str):
        """Handle changes in the entry fields."""
        try:
            value = float(event.widget.get())
            if "norm" in field_name and (value < 0 or value > 1):
                raise ValueError("Normalized values must be between 0 and 1")

            img_width, img_height = self.image.size

            if format_name == "VOC":
                new_box = self._update_voc_box(field_name, value)
            elif format_name == "COCO":
                new_box = self._update_coco_box(field_name, value)
            elif format_name == "YOLO":
                new_box = self._update_yolo_box(field_name, value)
            else:
                raise ValueError(f"Unknown format: {format_name}")

            self.bounding_box = BoundingBox(new_box, (img_height, img_width))
            self.update_entries()
            self.draw_bounding_box()

        except ValueError as e:
            tk.messagebox.showerror("Invalid Input", str(e))

    def _update_voc_box(self, field_name: str, value: float) -> VOCBox:
        """Update the VOC box with the new value."""
        current_box = self.bounding_box.to_voc(normalized="norm" in field_name)
        fields = ["xmin", "ymin", "xmax", "ymax"]
        index = fields.index(field_name.replace("_norm", ""))
        values = [current_box.xmin, current_box.ymin, current_box.xmax, current_box.ymax]
        values[index] = value
        return VOCBox(*values, normalized="norm" in field_name)

    def _update_coco_box(self, field_name: str, value: float) -> COCOBox:
        """Update the COCO box with the new value."""
        current_box = self.bounding_box.to_coco(normalized="norm" in field_name)
        fields = ["x", "y", "width", "height"]
        index = fields.index(field_name.replace("_norm", ""))
        values = [current_box.x, current_box.y, current_box.width, current_box.height]
        values[index] = value
        return COCOBox(*values, normalized="norm" in field_name)

    def _update_yolo_box(self, field_name: str, value: float) -> YOLOBox:
        """Update the YOLO box with the new value."""
        current_box = self.bounding_box.to_yolo(normalized="norm" in field_name)
        fields = ["x_center", "y_center", "width", "height"]
        index = fields.index(field_name.replace("_norm", ""))
        values = [current_box.x_center, current_box.y_center, current_box.width, current_box.height]
        values[index] = value
        return YOLOBox(*values, normalized="norm" in field_name)
