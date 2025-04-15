"""
UI Module
Contains the main application UI and logic.
"""
import os
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time

from .face_detection import FaceDetector
from .item_renderer import ItemRenderer
from .utils import (
    scan_assets_directory,
    get_available_styles,
    get_available_colors,
    convert_cv_to_tk,
    create_sample_assets
)


class VirtualTryOnApp:
    """Main Virtual Try-On application class."""

    def __init__(self, root, base_dir):
        """
        Initialize the application.

        Args:
            root: Tkinter root window
            base_dir: Base directory of the application
        """
        self.root = root
        self.base_dir = base_dir
        self.root.title("Virtual Try-On System")
        self.root.geometry("1200x700")
        self.root.minsize(1000, 600)

        # Set application icon if available
        icon_path = os.path.join(base_dir, "assets", "icon.png")
        if os.path.exists(icon_path):
            icon = ImageTk.PhotoImage(file=icon_path)
            self.root.iconphoto(True, icon)

        # Initialize components
        self.face_detector = FaceDetector()
        self.item_renderer = ItemRenderer(base_dir)

        # Initialize camera
        self.cap = None
        self.camera_active = False
        self.camera_thread = None
        self.camera_thread_stop = False

        # Initialize item types and colors
        self.item_types = ["Glasses", "Mask"]
        self.current_item_type = "Glasses"

        # Define default colors
        self.default_colors = ["Black", "Blue",
                               "Red", "Green", "White", "Yellow"]
        self.current_color = "Black"

        # Scan for available assets
        self.items = scan_assets_directory(base_dir)

        # If no assets found, create sample assets
        if (not self.items["Glasses"] and not self.items["Mask"]) or (
            all(len(styles) == 0 for styles in self.items["Glasses"].values()) and
            all(len(styles) == 0 for styles in self.items["Mask"].values())
        ):
            create_sample_assets(base_dir)
            self.items = scan_assets_directory(base_dir)

        # Get available styles
        self.available_styles = get_available_styles(
            self.items, self.current_item_type)
        self.current_item = self.available_styles[0] if self.available_styles else "Rectangle"

        # Get available colors
        self.available_colors = get_available_colors(
            self.items, self.current_item_type, self.current_item)
        self.current_color = self.available_colors[0] if self.available_colors else "Black"

        # Create UI
        self.create_ui()

        # Start camera
        self.start_camera()

    def create_ui(self):
        """Create the application UI."""
        # Create main frames
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main_frame = ttk.Frame(self.root)
        main_frame.grid(column=0, row=0, sticky=(
            tk.N, tk.W, tk.E, tk.S), padx=10, pady=10)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Control panel
        control_frame = ttk.Frame(
            main_frame, padding="10", relief="raised", borderwidth=1)
        control_frame.grid(column=0, row=0, sticky=(
            tk.N, tk.W, tk.S), padx=(0, 10))

        # Video frame
        video_frame = ttk.Frame(main_frame, padding="10")
        video_frame.grid(column=1, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)

        # Video display
        self.video_canvas = tk.Canvas(video_frame, bg="black")
        self.video_canvas.grid(
            column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(column=0, row=1, columnspan=2,
                          sticky=(tk.W, tk.E, tk.S), pady=(10, 0))
        status_frame.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="Status: Ready")
        status_label = ttk.Label(
            status_frame, textvariable=self.status_var, anchor=tk.W)
        status_label.grid(column=0, row=0, sticky=(tk.W, tk.E))

        # Camera control button
        self.camera_btn_var = tk.StringVar(value="Stop Camera")
        camera_btn = ttk.Button(status_frame, textvariable=self.camera_btn_var,
                                command=self.toggle_camera)
        camera_btn.grid(column=1, row=0, sticky=tk.E, padx=(10, 0))

        # Controls
        ttk.Label(control_frame, text="Virtual Try-On Controls",
                  font=("Arial", 16, "bold")).pack(pady=(0, 20))

        # Create control sections
        self.create_item_type_control(control_frame)
        self.create_item_style_control(control_frame)
        self.create_color_control(control_frame)
        self.create_instructions(control_frame)

    def create_item_type_control(self, parent):
        """Create item type selection controls."""
        section_frame = ttk.LabelFrame(
            parent, text="Item Type", padding=(10, 5))
        section_frame.pack(fill=tk.X, pady=(0, 10))

        self.item_type_var = tk.StringVar(value=self.current_item_type)

        for item_type in self.item_types:
            rb = ttk.Radiobutton(
                section_frame,
                text=item_type,
                value=item_type,
                variable=self.item_type_var,
                command=self.on_item_type_changed
            )
            rb.pack(anchor=tk.W, pady=2)

    def create_item_style_control(self, parent):
        """Create item style selection controls."""
        self.style_frame = ttk.LabelFrame(
            parent, text="Item Style", padding=(10, 5))
        self.style_frame.pack(fill=tk.X, pady=(0, 10))

        self.style_var = tk.StringVar(value=self.current_item)
        self.style_radios = []

        # Create radio buttons for styles
        self.update_style_radios()

    def update_style_radios(self):
        """Update the style radio buttons based on selected item type."""
        # Clear existing radio buttons
        for radio in self.style_radios:
            radio.destroy()
        self.style_radios = []

        # Create new radio buttons
        for style in self.available_styles:
            rb = ttk.Radiobutton(
                self.style_frame,
                text=style,
                value=style,
                variable=self.style_var,
                command=self.on_item_style_changed
            )
            rb.pack(anchor=tk.W, pady=2)
            self.style_radios.append(rb)

    def create_color_control(self, parent):
        """Create color selection controls."""
        color_section = ttk.LabelFrame(parent, text="Color", padding=(10, 5))
        color_section.pack(fill=tk.X, pady=(0, 10))

        # Color selection
        self.color_var = tk.StringVar(value=self.current_color)

        # Create a frame for the color radio buttons
        self.color_frame = ttk.Frame(color_section)
        self.color_frame.pack(fill=tk.X)

        self.color_radios = []

        # Update color radios
        self.update_color_radios()

        # Color preview
        preview_frame = ttk.Frame(color_section)
        preview_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(preview_frame, text="Preview:").pack(side=tk.LEFT)

        self.color_preview = tk.Canvas(
            preview_frame, width=80, height=20, bg='black')
        self.color_preview.pack(side=tk.LEFT, padx=(10, 0))

        # Update color preview
        self.update_color_preview()

    def update_color_radios(self):
        """Update the color radio buttons based on selected item style."""
        # Clear existing radio buttons
        for radio in self.color_radios:
            radio.destroy()
        self.color_radios = []

        # Create color radio buttons in a grid (3 columns)
        for i, color in enumerate(self.available_colors):
            row = i // 2
            col = i % 2

            rb = ttk.Radiobutton(
                self.color_frame,
                text=color,
                value=color,
                variable=self.color_var,
                command=self.on_color_changed
            )
            rb.grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
            self.color_radios.append(rb)

    def update_color_preview(self):
        """Update the color preview based on selected color."""
        # Get the RGB color
        if self.current_color in self.item_renderer.colors:
            bgr_color = self.item_renderer.colors[self.current_color]
            # Convert from BGR to RGB for tkinter
            rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
            hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_color)
            self.color_preview.config(bg=hex_color)

    def create_instructions(self, parent):
        """Create instructions section."""
        instr_frame = ttk.LabelFrame(
            parent, text="Instructions", padding=(10, 5))
        instr_frame.pack(fill=tk.X, pady=(0, 10))

        instructions = """
1. Select an item type (Glasses/Mask)
2. Choose a style
3. Pick a color
4. Look at the camera

The virtual item will be overlaid on your face in real-time.

Add your own PNG images to the assets folder:
assets/head_fashions/glasses/[style]/[color].png
assets/head_fashions/masks/[style]/[color].png
        """

        instr_label = ttk.Label(instr_frame, text=instructions,
                                justify=tk.LEFT, wraplength=200)
        instr_label.pack(anchor=tk.W, pady=5)

        # Add version info
        from .init import __version__
        version_label = ttk.Label(parent, text=f"Version {__version__}",
                                  font=("Arial", 8), foreground="gray")
        version_label.pack(side=tk.BOTTOM, anchor=tk.E, pady=(10, 0))

    def on_item_type_changed(self):
        """Handle item type change."""
        new_type = self.item_type_var.get()
        if new_type != self.current_item_type:
            self.current_item_type = new_type

            # Update available styles
            self.available_styles = get_available_styles(
                self.items, self.current_item_type)

            # Update current style
            if self.available_styles:
                self.current_item = self.available_styles[0]
                self.style_var.set(self.current_item)

            # Update style radio buttons
            self.update_style_radios()

            # Update available colors
            self.available_colors = get_available_colors(
                self.items, self.current_item_type, self.current_item
            )

            # Update current color
            if self.available_colors:
                self.current_color = self.available_colors[0]
                self.color_var.set(self.current_color)

            # Update color radio buttons
            self.update_color_radios()

            # Update color preview
            self.update_color_preview()

    def on_item_style_changed(self):
        """Handle item style change."""
        new_style = self.style_var.get()
        if new_style != self.current_item:
            self.current_item = new_style

            # Update available colors
            self.available_colors = get_available_colors(
                self.items, self.current_item_type, self.current_item
            )

            # Update current color
            if self.available_colors:
                self.current_color = self.available_colors[0]
                self.color_var.set(self.current_color)

            # Update color radio buttons
            self.update_color_radios()

            # Update color preview
            self.update_color_preview()

    def on_color_changed(self):
        """Handle color change."""
        new_color = self.color_var.get()
        if new_color != self.current_color:
            self.current_color = new_color
            self.update_color_preview()

    def start_camera(self):
        """Start the camera."""
        if self.camera_active:
            return

        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera.")
                return

            self.camera_active = True
            self.camera_thread_stop = False
            self.camera_thread = threading.Thread(target=self.camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()

            self.status_var.set("Status: Camera running")
            self.camera_btn_var.set("Stop Camera")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")

    def stop_camera(self):
        """Stop the camera."""
        if not self.camera_active:
            return

        self.camera_thread_stop = True
        if self.camera_thread:
            self.camera_thread.join(timeout=1.0)

        if self.cap:
            self.cap.release()
            self.cap = None

        self.camera_active = False
        self.status_var.set("Status: Camera stopped")
        self.camera_btn_var.set("Start Camera")

        # Clear the video display
        self.video_canvas.delete("all")
        self.video_canvas.create_text(
            self.video_canvas.winfo_width() // 2,
            self.video_canvas.winfo_height() // 2,
            text="Camera Off",
            fill="white",
            font=("Arial", 24)
        )

    def toggle_camera(self):
        """Toggle the camera on/off."""
        if self.camera_active:
            self.stop_camera()
        else:
            self.start_camera()

    def camera_loop(self):
        """Main camera processing loop."""
        last_time = time.time()
        fps_counter = 0
        fps = 0

        # Store the last successful frame for smoothness
        last_frame = None

        # Face detection smoothing
        last_face_data = None
        face_detection_cooldown = 0

        while not self.camera_thread_stop:
            try:
                # Calculate FPS
                current_time = time.time()
                fps_counter += 1

                if current_time - last_time >= 1.0:
                    fps = fps_counter
                    fps_counter = 0
                    last_time = current_time

                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    # If frame reading failed but we have a previous frame, use it
                    if last_frame is not None:
                        frame = last_frame.copy()
                    else:
                        self.status_var.set("Status: Camera error")
                        time.sleep(0.1)  # Small delay to prevent CPU spike
                        continue
                else:
                    # Store successful frame for backup
                    last_frame = frame.copy()

                # Flip the frame horizontally for selfie view
                frame = cv2.flip(frame, 1)

                # Face detection with cooldown to reduce flickering
                face_data = None

                if face_detection_cooldown <= 0:
                    # Detect face
                    landmarks = self.face_detector.detect_face(frame)

                    # If face detected, get face data
                    if landmarks:
                        img_h, img_w = frame.shape[:2]
                        face_data = self.face_detector.get_face_dimensions(
                            landmarks, img_w, img_h)
                        last_face_data = face_data
                        face_detection_cooldown = 2  # Skip next 2 frames for detection
                    else:
                        # If no face detected this frame but we had one recently, keep using it
                        if last_face_data and face_detection_cooldown <= 5:  # Only reuse for a short while
                            face_data = last_face_data
                else:
                    # Use previous face data during cooldown
                    face_data = last_face_data
                    face_detection_cooldown -= 1

                # Render the selected item if we have face data
                if face_data:
                    frame = self.item_renderer.render_item(
                        frame, face_data, self.current_item_type,
                        self.current_item, self.current_color
                    )

                # Add FPS counter
                cv2.putText(
                    frame, f"FPS: {fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )

                # Convert to PhotoImage and update canvas - use main thread
                # to avoid tkinter threading issues
                self.root.after(
                    0, lambda f=frame: self.update_video_display(f))

                # Give some time to the UI thread to update
                time.sleep(0.01)

            except Exception as e:
                self.status_var.set(f"Error: {str(e)}")
                time.sleep(0.1)  # Small delay to prevent CPU spike

    def update_video_display(self, frame):
        """Update the video display with the current frame."""
        # Get canvas dimensions
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not yet properly initialized
            return

        try:
            # Resize frame to fit canvas while maintaining aspect ratio
            img_h, img_w = frame.shape[:2]

            # Calculate scaling factor
            scale_w = canvas_width / img_w
            scale_h = canvas_height / img_h
            scale = min(scale_w, scale_h)

            # Calculate new dimensions
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)

            # Resize frame
            frame_resized = cv2.resize(frame, (new_w, new_h))

            # Convert to PhotoImage
            photo = convert_cv_to_tk(frame_resized)

            # Update canvas
            self.video_canvas.delete("all")
            self.video_canvas.create_image(
                canvas_width // 2, canvas_height // 2,
                image=photo, anchor=tk.CENTER
            )

            # Keep a reference to prevent garbage collection
            self.video_canvas.photo = photo

        except Exception as e:
            print(f"Error updating display: {str(e)}")

    def on_closing(self):
        """Handle window closing."""
        # Stop camera
        self.stop_camera()

        # Release resources
        self.face_detector.release()

        # Destroy window
        self.root.destroy()
