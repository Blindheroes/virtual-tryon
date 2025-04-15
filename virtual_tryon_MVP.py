import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import glob


class VirtualTryOnApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Virtual Try-On System")
        self.root.geometry("1200x700")

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize item types and colors
        self.item_types = ["Glasses", "Mask"]
        self.current_item_type = "Glasses"

        # Define colors for items
        self.colors = {
            "Red": (0, 0, 255),
            "Green": (0, 255, 0),
            "Blue": (255, 0, 0),
            "Black": (0, 0, 0),
            "White": (255, 255, 255),
            "Yellow": (0, 255, 255)
        }
        self.current_color = "Black"

        # Load sample item images
        self.load_items()

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)

        # Create UI
        self.create_ui()

        # Start video loop
        self.update_frame()

    def load_items(self):
        # In a real application, you would load actual transparent PNG images
        # For the MVP, we'll create simple shapes for glasses and masks

        # Initialize items dictionary
        self.items = {
            "Glasses": {
                "Rectangle": None,
                "Round": None,
                "Aviator": None
            },
            "Mask": {
                "Medical": None,
                "N95": None,
                "Fashion": None
            }
        }

        # In a real application, we would load PNG files with transparency
        # For now, we'll draw the items dynamically

        self.current_item = list(self.items[self.current_item_type].keys())[0]

    def create_ui(self):
        # Create main frames
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        video_frame = ttk.Frame(self.root, padding="10")
        video_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Video display
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(expand=True, fill=tk.BOTH)

        # Controls
        ttk.Label(control_frame, text="Virtual Try-On Controls",
                  font=("Arial", 16)).pack(pady=10)

        # Item type selection
        ttk.Label(control_frame, text="Item Type:").pack(
            anchor=tk.W, pady=(10, 5))
        self.item_type_var = tk.StringVar(value=self.current_item_type)
        item_type_cb = ttk.Combobox(control_frame, textvariable=self.item_type_var,
                                    values=self.item_types, state="readonly", width=15)
        item_type_cb.pack(anchor=tk.W, pady=(0, 10))
        item_type_cb.bind("<<ComboboxSelected>>", self.on_item_type_changed)

        # Item selection
        ttk.Label(control_frame, text="Item Style:").pack(
            anchor=tk.W, pady=(10, 5))
        self.item_var = tk.StringVar(value=self.current_item)
        self.item_cb = ttk.Combobox(control_frame, textvariable=self.item_var,
                                    values=list(
                                        self.items[self.current_item_type].keys()),
                                    state="readonly", width=15)
        self.item_cb.pack(anchor=tk.W, pady=(0, 10))
        self.item_cb.bind("<<ComboboxSelected>>", self.on_item_changed)

        # Color selection
        ttk.Label(control_frame, text="Color:").pack(anchor=tk.W, pady=(10, 5))
        self.color_var = tk.StringVar(value=self.current_color)
        color_cb = ttk.Combobox(control_frame, textvariable=self.color_var,
                                values=list(self.colors.keys()),
                                state="readonly", width=15)
        color_cb.pack(anchor=tk.W, pady=(0, 10))
        color_cb.bind("<<ComboboxSelected>>", self.on_color_changed)

        # Color preview
        self.color_preview = tk.Canvas(
            control_frame, width=100, height=30, bg='black')
        self.color_preview.pack(anchor=tk.W, pady=(0, 20))

        # Instructions
        instructions = """
        Instructions:
        1. Select an item type (Glasses/Mask)
        2. Choose a style
        3. Pick a color
        4. Look at the camera
        5. The virtual item will overlay on your face
        """
        ttk.Label(control_frame, text=instructions, wraplength=200,
                  justify=tk.LEFT).pack(anchor=tk.W, pady=10)

    def on_item_type_changed(self, event):
        self.current_item_type = self.item_type_var.get()
        self.current_item = list(self.items[self.current_item_type].keys())[0]
        self.item_var.set(self.current_item)
        self.item_cb.config(values=list(
            self.items[self.current_item_type].keys()))

    def on_item_changed(self, event):
        self.current_item = self.item_var.get()

    def on_color_changed(self, event):
        self.current_color = self.color_var.get()
        rgb_color = self.colors[self.current_color]
        # Convert from BGR to RGB for tkinter
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            rgb_color[2], rgb_color[1], rgb_color[0])
        self.color_preview.config(bg=hex_color)

    def draw_glasses(self, frame, landmarks):
        """Draw glasses on the face."""
        if landmarks is None:
            return frame

        img_h, img_w = frame.shape[:2]

        # Get key points for glasses placement
        # Left eye, right eye, nose, left ear, right ear
        left_eye = landmarks[33]  # Left eye outer corner
        right_eye = landmarks[263]  # Right eye outer corner
        nose_tip = landmarks[4]  # Nose tip
        left_temple = landmarks[162]  # Left temple area
        right_temple = landmarks[389]  # Right temple area

        # Convert normalized coordinates to pixel coordinates
        left_eye_px = (int(left_eye.x * img_w), int(left_eye.y * img_h))
        right_eye_px = (int(right_eye.x * img_w), int(right_eye.y * img_h))
        nose_tip_px = (int(nose_tip.x * img_w), int(nose_tip.y * img_h))
        left_temple_px = (int(left_temple.x * img_w),
                          int(left_temple.y * img_h))
        right_temple_px = (int(right_temple.x * img_w),
                           int(right_temple.y * img_h))

        # Calculate glasses dimensions based on face landmarks
        glasses_width = int(1.1 * abs(right_eye_px[0] - left_eye_px[0]))
        glasses_height = int(0.5 * glasses_width)

        # Calculate the center point between the eyes
        eye_center_x = (left_eye_px[0] + right_eye_px[0]) // 2
        eye_center_y = (left_eye_px[1] + right_eye_px[1]) // 2

        # Create an overlay for the glasses
        overlay = frame.copy()

        # Choose glasses style based on selection
        color = self.colors[self.current_color]

        if self.current_item == "Rectangle":
            # Draw rectangular glasses
            left_lens_pos = (eye_center_x - glasses_width // 2, eye_center_y)
            right_lens_pos = (eye_center_x + glasses_width // 2, eye_center_y)

            # Draw glasses frame
            cv2.rectangle(overlay,
                          (left_lens_pos[0] - glasses_width // 4,
                           left_lens_pos[1] - glasses_height // 2),
                          (left_lens_pos[0] + glasses_width // 4,
                           left_lens_pos[1] + glasses_height // 2),
                          color, 2)

            cv2.rectangle(overlay,
                          (right_lens_pos[0] - glasses_width // 4,
                           right_lens_pos[1] - glasses_height // 2),
                          (right_lens_pos[0] + glasses_width // 4,
                           right_lens_pos[1] + glasses_height // 2),
                          color, 2)

            # Draw bridge
            cv2.line(overlay,
                     (left_lens_pos[0] + glasses_width // 4, eye_center_y),
                     (right_lens_pos[0] - glasses_width // 4, eye_center_y),
                     color, 2)

            # Draw temples (arms)
            temple_length = glasses_width // 2
            cv2.line(overlay,
                     (left_lens_pos[0] - glasses_width // 4, eye_center_y),
                     (left_lens_pos[0] - glasses_width //
                      4 - temple_length, eye_center_y),
                     color, 2)

            cv2.line(overlay,
                     (right_lens_pos[0] + glasses_width // 4, eye_center_y),
                     (right_lens_pos[0] + glasses_width //
                      4 + temple_length, eye_center_y),
                     color, 2)

        elif self.current_item == "Round":
            # Draw round glasses
            lens_radius = glasses_width // 4

            # Draw lenses
            cv2.circle(overlay,
                       (eye_center_x - glasses_width // 4, eye_center_y),
                       lens_radius, color, 2)

            cv2.circle(overlay,
                       (eye_center_x + glasses_width // 4, eye_center_y),
                       lens_radius, color, 2)

            # Draw bridge
            cv2.line(overlay,
                     (eye_center_x - glasses_width //
                      4 + lens_radius, eye_center_y),
                     (eye_center_x + glasses_width //
                      4 - lens_radius, eye_center_y),
                     color, 2)

            # Draw temples (arms)
            temple_length = glasses_width // 2
            cv2.line(overlay,
                     (eye_center_x - glasses_width //
                      4 - lens_radius, eye_center_y),
                     (eye_center_x - glasses_width // 4 - lens_radius -
                      temple_length, eye_center_y - temple_length // 4),
                     color, 2)

            cv2.line(overlay,
                     (eye_center_x + glasses_width //
                      4 + lens_radius, eye_center_y),
                     (eye_center_x + glasses_width // 4 + lens_radius +
                      temple_length, eye_center_y - temple_length // 4),
                     color, 2)

        elif self.current_item == "Aviator":
            # Draw aviator glasses
            # Draw lenses with a teardrop shape

            # Left lens
            left_center = (eye_center_x - glasses_width // 4, eye_center_y)
            left_points = np.array([
                [left_center[0] - glasses_width // 6,
                    left_center[1] - glasses_height // 3],
                [left_center[0] + glasses_width // 6,
                    left_center[1] - glasses_height // 3],
                [left_center[0] + glasses_width // 6, left_center[1]],
                [left_center[0], left_center[1] + glasses_height // 3],
                [left_center[0] - glasses_width // 6, left_center[1]]
            ], np.int32)
            left_points = left_points.reshape((-1, 1, 2))
            cv2.polylines(overlay, [left_points], True, color, 2)

            # Right lens
            right_center = (eye_center_x + glasses_width // 4, eye_center_y)
            right_points = np.array([
                [right_center[0] - glasses_width // 6,
                    right_center[1] - glasses_height // 3],
                [right_center[0] + glasses_width // 6,
                    right_center[1] - glasses_height // 3],
                [right_center[0] + glasses_width // 6, right_center[1]],
                [right_center[0], right_center[1] + glasses_height // 3],
                [right_center[0] - glasses_width // 6, right_center[1]]
            ], np.int32)
            right_points = right_points.reshape((-1, 1, 2))
            cv2.polylines(overlay, [right_points], True, color, 2)

            # Draw bridge
            cv2.line(overlay,
                     (left_center[0] + glasses_width // 6,
                      left_center[1] - glasses_height // 6),
                     (right_center[0] - glasses_width // 6,
                      right_center[1] - glasses_height // 6),
                     color, 2)

            # Draw temples (arms)
            temple_length = glasses_width // 2
            cv2.line(overlay,
                     (left_center[0] - glasses_width // 6,
                      left_center[1] - glasses_height // 4),
                     (left_center[0] - glasses_width //
                      6 - temple_length, left_center[1]),
                     color, 2)

            cv2.line(overlay,
                     (right_center[0] + glasses_width // 6,
                      right_center[1] - glasses_height // 4),
                     (right_center[0] + glasses_width //
                      6 + temple_length, right_center[1]),
                     color, 2)

        # Apply the overlay with transparency
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame

    def draw_mask(self, frame, landmarks):
        """Draw mask on the face."""
        if landmarks is None:
            return frame

        img_h, img_w = frame.shape[:2]

        # Get key points for mask placement
        # Face contour points
        face_contour_points = [
            landmarks[152],  # Left cheek
            landmarks[10],   # Chin
            landmarks[378]   # Right cheek
        ]

        nose_bridge = [landmarks[6], landmarks[197]]  # Nose bridge points

        # Convert normalized coordinates to pixel coordinates
        face_contour_px = [(int(pt.x * img_w), int(pt.y * img_h))
                           for pt in face_contour_points]
        nose_bridge_px = [(int(pt.x * img_w), int(pt.y * img_h))
                          for pt in nose_bridge]

        # Create an overlay for the mask
        overlay = frame.copy()

        # Choose mask style based on selection
        color = self.colors[self.current_color]

        if self.current_item == "Medical":
            # Draw a medical-style mask
            # Get additional points for mask shape
            left_face = landmarks[234]
            right_face = landmarks[454]
            chin = landmarks[152]

            left_face_px = (int(left_face.x * img_w), int(left_face.y * img_h))
            right_face_px = (int(right_face.x * img_w),
                             int(right_face.y * img_h))
            chin_px = (int(chin.x * img_w), int(chin.y * img_h))

            # Create polygon points for mask
            mask_points = np.array([
                [left_face_px[0], left_face_px[1]],
                [nose_bridge_px[0][0] - 20, nose_bridge_px[0][1]],
                [nose_bridge_px[1][0] + 20, nose_bridge_px[1][1]],
                [right_face_px[0], right_face_px[1]],
                [face_contour_px[2][0], face_contour_px[2][1] + 20],
                [face_contour_px[1][0], face_contour_px[1][1] + 10],
                [face_contour_px[0][0], face_contour_px[0][1] + 20]
            ], np.int32)

            mask_points = mask_points.reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [mask_points], color)

            # Draw mask bands
            ear_left = landmarks[234]
            ear_right = landmarks[454]
            ear_left_px = (int(ear_left.x * img_w), int(ear_left.y * img_h))
            ear_right_px = (int(ear_right.x * img_w), int(ear_right.y * img_h))

            cv2.line(overlay, (left_face_px[0], left_face_px[1]),
                     (ear_left_px[0] - 30, ear_left_px[1]), (70, 70, 70), 3)
            cv2.line(overlay, (right_face_px[0], right_face_px[1]),
                     (ear_right_px[0] + 30, ear_right_px[1]), (70, 70, 70), 3)

        elif self.current_item == "N95":
            # Draw an N95-style mask
            # Get additional points for mask shape
            nose_tip = landmarks[4]
            left_cheek = landmarks[234]
            right_cheek = landmarks[454]
            chin = landmarks[152]

            nose_tip_px = (int(nose_tip.x * img_w), int(nose_tip.y * img_h))
            left_cheek_px = (int(left_cheek.x * img_w),
                             int(left_cheek.y * img_h))
            right_cheek_px = (int(right_cheek.x * img_w),
                              int(right_cheek.y * img_h))
            chin_px = (int(chin.x * img_w), int(chin.y * img_h))

            # Create polygon points for mask
            mask_points = np.array([
                [left_cheek_px[0], left_cheek_px[1]],
                [nose_bridge_px[0][0] - 15, nose_bridge_px[0][1]],
                [nose_bridge_px[1][0] + 15, nose_bridge_px[1][1]],
                [right_cheek_px[0], right_cheek_px[1]],
                [face_contour_px[2][0], face_contour_px[2][1] + 30],
                [face_contour_px[1][0], face_contour_px[1][1] + 15],
                [face_contour_px[0][0], face_contour_px[0][1] + 30]
            ], np.int32)

            mask_points = mask_points.reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [mask_points], color)

            # Draw mask edges with a different color
            cv2.polylines(overlay, [mask_points], True, (70, 70, 70), 2)

            # Draw a small valve circle
            valve_pos = (nose_tip_px[0], nose_tip_px[1] + 20)
            cv2.circle(overlay, valve_pos, 8, (70, 70, 70), -1)

            # Draw mask bands
            ear_left = landmarks[234]
            ear_right = landmarks[454]
            ear_left_px = (int(ear_left.x * img_w), int(ear_left.y * img_h))
            ear_right_px = (int(ear_right.x * img_w), int(ear_right.y * img_h))

            # Top band
            cv2.line(overlay, (left_cheek_px[0], left_cheek_px[1] - 10),
                     (ear_left_px[0] - 40, ear_left_px[1] - 30), (200, 200, 200), 3)
            cv2.line(overlay, (right_cheek_px[0], right_cheek_px[1] - 10),
                     (ear_right_px[0] + 40, ear_right_px[1] - 30), (200, 200, 200), 3)

            # Bottom band
            cv2.line(overlay, (left_cheek_px[0], left_cheek_px[1] + 15),
                     (ear_left_px[0] - 40, ear_left_px[1] + 20), (200, 200, 200), 3)
            cv2.line(overlay, (right_cheek_px[0], right_cheek_px[1] + 15),
                     (ear_right_px[0] + 40, ear_right_px[1] + 20), (200, 200, 200), 3)

        elif self.current_item == "Fashion":
            # Draw a fashion-style mask
            mask_height = int(
                abs(face_contour_px[1][1] - nose_bridge_px[0][1]) * 1.3)
            mask_width = int(
                abs(face_contour_px[0][0] - face_contour_px[2][0]) * 1.1)

            # Center point
            center_x = (face_contour_px[0][0] + face_contour_px[2][0]) // 2
            center_y = (face_contour_px[0][1] + nose_bridge_px[0][1]) // 2

            # Create polygon points for mask
            mask_points = np.array([
                [center_x - mask_width // 2, nose_bridge_px[0][1]],
                [center_x + mask_width // 2, nose_bridge_px[0][1]],
                [face_contour_px[2][0] + 10, face_contour_px[2][1] + 10],
                [face_contour_px[1][0], face_contour_px[1][1] + 15],
                [face_contour_px[0][0] - 10, face_contour_px[0][1] + 10]
            ], np.int32)

            mask_points = mask_points.reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [mask_points], color)

            # Add pattern (dots)
            for i in range(0, mask_width, 20):
                for j in range(0, mask_height, 20):
                    dot_x = center_x - mask_width // 2 + i
                    dot_y = nose_bridge_px[0][1] + j

                    # Check if the point is inside the mask
                    if cv2.pointPolygonTest(mask_points, (dot_x, dot_y), False) >= 0:
                        # Alternate color for pattern
                        pattern_color = (
                            255, 255, 255) if self.current_color != "White" else (0, 0, 0)
                        cv2.circle(overlay, (dot_x, dot_y),
                                   3, pattern_color, -1)

            # Add mask edges
            cv2.polylines(overlay, [mask_points], True, (70, 70, 70), 2)

            # Draw earloops
            ear_left = landmarks[234]
            ear_right = landmarks[454]
            ear_left_px = (int(ear_left.x * img_w), int(ear_left.y * img_h))
            ear_right_px = (int(ear_right.x * img_w), int(ear_right.y * img_h))

            cv2.line(overlay, (center_x - mask_width // 2, nose_bridge_px[0][1] + 10),
                     (ear_left_px[0] - 20, ear_left_px[1]), (70, 70, 70), 2)
            cv2.line(overlay, (center_x + mask_width // 2, nose_bridge_px[0][1] + 10),
                     (ear_right_px[0] + 20, ear_right_px[1]), (70, 70, 70), 2)

        # Apply the overlay with transparency
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Flip the frame horizontally for selfie view
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image with MediaPipe Face Mesh
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark

                # Apply virtual try-on effect based on selection
                if self.current_item_type == "Glasses":
                    frame = self.draw_glasses(frame, face_landmarks)
                elif self.current_item_type == "Mask":
                    frame = self.draw_mask(frame, face_landmarks)

            # Convert BGR to RGB for Tkinter
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        # Update frame every 33ms (approx 30 FPS)
        self.root.after(33, self.update_frame)

    def on_closing(self):
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = VirtualTryOnApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
    
