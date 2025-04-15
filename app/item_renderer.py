"""
Item Renderer Module
Handles the rendering of virtual items on the face.
"""
import os
import cv2
import numpy as np


class ItemRenderer:
    """Renders virtual items on a face image."""

    def __init__(self, base_dir):
        """
        Initialize the item renderer.

        Args:
            base_dir: Base directory of the application
        """
        self.base_dir = base_dir
        self.items_cache = {}  # Cache for loaded images

        # BGR color values
        self.colors = {
            "Red": (0, 0, 255),
            "Green": (0, 255, 0),
            "Blue": (255, 0, 0),
            "Black": (0, 0, 0),
            "White": (255, 255, 255),
            "Yellow": (0, 255, 255)
        }

    def load_item_image(self, item_type, style, color):
        """
        Load an item image from the assets folder.

        Args:
            item_type: Type of item (glasses or masks)
            style: Style of the item
            color: Color of the item

        Returns:
            The loaded image or None if not found
        """
        # Create cache key
        cache_key = f"{item_type}_{style}_{color}"

        # Check if image is already in cache
        if cache_key in self.items_cache:
            return self.items_cache[cache_key]

        # Determine file path
        folder = "glasses" if item_type == "Glasses" else "masks"
        style_lower = style.lower()
        color_lower = color.lower()

        # Try to find the image with either .png or .PNG extension
        img_path = os.path.join(self.base_dir, "assets", "head_fashions",
                                folder, style_lower, f"{color_lower}.png")

        if os.path.exists(img_path):
            # Load the image with alpha channel
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            self.items_cache[cache_key] = img
            return img

        # If image doesn't exist, return None
        return None

    def render_item(self, frame, face_data, item_type, style, color):
        """
        Render a virtual item on the face.

        Args:
            frame: BGR frame from the camera
            face_data: Face dimensions and landmarks
            item_type: Type of item (Glasses or Mask)
            style: Style of the item
            color: Color of the item

        Returns:
            Frame with the rendered item
        """
        if face_data is None:
            return frame

        # Try to load the image
        img = self.load_item_image(item_type, style, color)

        # If image exists, overlay it
        if img is not None:
            return self._overlay_image(frame, img, face_data, item_type, style)

        # If no image, fall back to drawn items
        if item_type == "Glasses":
            return self._draw_glasses(frame, face_data, style, color)
        else:  # Mask
            return self._draw_mask(frame, face_data, style, color)

    def _overlay_image(self, frame, img, face_data, item_type, style):
        """
        Overlay an image on the frame based on face position.

        Args:
            frame: BGR frame from the camera
            img: Item image with alpha channel
            face_data: Face dimensions and landmarks
            item_type: Type of item
            style: Style of the item

        Returns:
            Frame with the overlaid image
        """
        try:
            # Get dimensions
            frame_h, frame_w = frame.shape[:2]
            if img is None or img.size == 0:
                return frame

            img_h, img_w = img.shape[:2]

            # Scale and position based on item type
            if item_type == "Glasses":
                points = face_data["points"]
                dimensions = face_data["dimensions"]

                # Calculate scale based on face width
                face_width = abs(points["right_eye"]
                                 [0] - points["left_eye"][0]) * 2.0
                scale_factor = face_width / img_w

                # Resize the image
                new_width = int(img_w * scale_factor)
                new_height = int(img_h * scale_factor)

                # Additional adjustment for aviator style
                y_offset_adjust = 0
                if style == "Aviator":
                    y_offset_adjust = int(new_height * 0.1)

                # Calculate position (centered between eyes)
                x_offset = int(dimensions["eye_center"][0] - new_width / 2)
                y_offset = int(points["nose_bridge_top"]
                               [1] - new_height / 3) + y_offset_adjust

            else:  # Mask
                points = face_data["points"]

                # Calculate scale based on face width
                face_width = abs(points["right_cheek"]
                                 [0] - points["left_cheek"][0]) * 2.5
                scale_factor = face_width / img_w

                # Resize the image
                new_width = int(img_w * scale_factor)
                new_height = int(img_h * scale_factor)

                # Calculate position (centered on nose)
                x_offset = int(points["nose_tip"][0] - new_width / 2)
                y_offset = int(points["nose_tip"][1] - new_height / 3)

            # Ensure we can resize the image
            if new_width <= 0 or new_height <= 0:
                return frame

            # Safely resize the image
            img_resized = cv2.resize(
                img, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Ensure offsets are within frame
            x_offset = max(0, min(frame_w - 1, x_offset))
            y_offset = max(0, min(frame_h - 1, y_offset))

            # Apply overlay with alpha blending
            if img_resized.shape[2] == 4:  # Has alpha channel
                # Calculate region to use (prevent out-of-bounds)
                y_end = min(y_offset + new_height, frame_h)
                x_end = min(x_offset + new_width, frame_w)

                # Handle edge cases where the item would go out of bounds
                if y_end <= y_offset or x_end <= x_offset:
                    return frame

                # Calculate height and width to use
                h_to_use = y_end - y_offset
                w_to_use = x_end - x_offset

                # Check if dimensions match
                if h_to_use <= 0 or w_to_use <= 0:
                    return frame

                # Extract the alpha channel for the visible portion
                alpha_s = img_resized[:h_to_use, :w_to_use, 3] / 255.0
                alpha_l = 1.0 - alpha_s

                # Create copy to avoid modifying the original frame
                result = frame.copy()

                # For each color channel
                for c in range(3):
                    # Apply the overlay
                    result[y_offset:y_end, x_offset:x_end, c] = (
                        alpha_s * img_resized[:h_to_use, :w_to_use, c] +
                        alpha_l * frame[y_offset:y_end, x_offset:x_end, c]
                    )

                return result

        except Exception as e:
            print(f"Overlay error: {str(e)}")
            # Return original frame if any error occurs
            return frame

        return frame

    def _draw_glasses(self, frame, face_data, style, color):
        """
        Draw glasses on the face using OpenCV.

        Args:
            frame: BGR frame from the camera
            face_data: Face dimensions and landmarks
            style: Style of glasses
            color: Color of glasses

        Returns:
            Frame with drawn glasses
        """
        points = face_data["points"]
        dimensions = face_data["dimensions"]

        eye_center_x = dimensions["eye_center"][0]
        eye_center_y = dimensions["eye_center"][1]

        glasses_width = int(
            1.1 * abs(points["right_eye"][0] - points["left_eye"][0]))
        glasses_height = int(0.5 * glasses_width)

        # Create an overlay for the glasses
        overlay = frame.copy()
        color_bgr = self.colors[color]

        if style == "Rectangle":
            # Draw rectangular glasses
            left_lens_pos = (eye_center_x - glasses_width // 2, eye_center_y)
            right_lens_pos = (eye_center_x + glasses_width // 2, eye_center_y)

            # Draw glasses frame
            cv2.rectangle(overlay,
                          (left_lens_pos[0] - glasses_width // 4,
                           left_lens_pos[1] - glasses_height // 2),
                          (left_lens_pos[0] + glasses_width // 4,
                              left_lens_pos[1] + glasses_height // 2),
                          color_bgr, 2)

            cv2.rectangle(overlay,
                          (right_lens_pos[0] - glasses_width // 4,
                           right_lens_pos[1] - glasses_height // 2),
                          (right_lens_pos[0] + glasses_width // 4,
                              right_lens_pos[1] + glasses_height // 2),
                          color_bgr, 2)

            # Draw bridge
            cv2.line(overlay,
                     (left_lens_pos[0] + glasses_width // 4, eye_center_y),
                     (right_lens_pos[0] - glasses_width // 4, eye_center_y),
                     color_bgr, 2)

            # Draw temples (arms)
            temple_length = glasses_width // 2
            cv2.line(overlay,
                     (left_lens_pos[0] - glasses_width // 4, eye_center_y),
                     (left_lens_pos[0] - glasses_width //
                      4 - temple_length, eye_center_y),
                     color_bgr, 2)

            cv2.line(overlay,
                     (right_lens_pos[0] + glasses_width // 4, eye_center_y),
                     (right_lens_pos[0] + glasses_width //
                      4 + temple_length, eye_center_y),
                     color_bgr, 2)

        elif style == "Round":
            # Draw round glasses
            lens_radius = glasses_width // 4

            # Draw lenses
            cv2.circle(overlay,
                       (eye_center_x - glasses_width // 4, eye_center_y),
                       lens_radius, color_bgr, 2)

            cv2.circle(overlay,
                       (eye_center_x + glasses_width // 4, eye_center_y),
                       lens_radius, color_bgr, 2)

            # Draw bridge
            cv2.line(overlay,
                     (eye_center_x - glasses_width //
                      4 + lens_radius, eye_center_y),
                     (eye_center_x + glasses_width //
                      4 - lens_radius, eye_center_y),
                     color_bgr, 2)

            # Draw temples (arms)
            temple_length = glasses_width // 2
            cv2.line(overlay,
                     (eye_center_x - glasses_width //
                      4 - lens_radius, eye_center_y),
                     (eye_center_x - glasses_width // 4 - lens_radius - temple_length,
                         eye_center_y - temple_length // 4),
                     color_bgr, 2)

            cv2.line(overlay,
                     (eye_center_x + glasses_width //
                      4 + lens_radius, eye_center_y),
                     (eye_center_x + glasses_width // 4 + lens_radius + temple_length,
                         eye_center_y - temple_length // 4),
                     color_bgr, 2)

        elif style == "Aviator":
            # Draw aviator glasses
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
            cv2.polylines(overlay, [left_points], True, color_bgr, 2)

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
            cv2.polylines(overlay, [right_points], True, color_bgr, 2)

            # Draw bridge
            cv2.line(overlay,
                     (left_center[0] + glasses_width // 6,
                      left_center[1] - glasses_height // 6),
                     (right_center[0] - glasses_width // 6,
                      right_center[1] - glasses_height // 6),
                     color_bgr, 2)

            # Draw temples (arms)
            temple_length = glasses_width // 2
            cv2.line(overlay,
                     (left_center[0] - glasses_width // 6,
                      left_center[1] - glasses_height // 4),
                     (left_center[0] - glasses_width //
                      6 - temple_length, left_center[1]),
                     color_bgr, 2)

            cv2.line(overlay,
                     (right_center[0] + glasses_width // 6,
                      right_center[1] - glasses_height // 4),
                     (right_center[0] + glasses_width //
                      6 + temple_length, right_center[1]),
                     color_bgr, 2)

        # Apply the overlay with transparency
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame

    def _draw_mask(self, frame, face_data, style, color):
        """
        Draw a mask on the face using OpenCV.

        Args:
            frame: BGR frame from the camera
            face_data: Face dimensions and landmarks
            style: Style of mask
            color: Color of mask

        Returns:
            Frame with drawn mask
        """
        points = face_data["points"]
        dimensions = face_data["dimensions"]

        # Create an overlay for the mask
        overlay = frame.copy()
        color_bgr = self.colors[color]

        # Increase the size of all masks
        mask_scale_factor = 1.5  # Adjust this value to make masks larger

        if style == "Medical":
            # Get additional points for mask shape
            left_face = points["left_cheek"]
            right_face = points["right_cheek"]
            chin = points["chin"]
            nose_bridge_top = points["nose_bridge_top"]
            nose_bridge_bottom = points["nose_bridge_bottom"]

            # Adjust mask width by scaling outward from center
            center_x = (left_face[0] + right_face[0]) // 2
            left_x = center_x - \
                int((center_x - left_face[0]) * mask_scale_factor)
            right_x = center_x + \
                int((right_face[0] - center_x) * mask_scale_factor)

            # Create polygon points for mask
            mask_points = np.array([
                [left_face[0], left_face[1]],
                [nose_bridge_top[0] - 20, nose_bridge_top[1]],
                [nose_bridge_bottom[0] + 20, nose_bridge_bottom[1]],
                [right_face[0], right_face[1]],
                [points["right_cheek"][0], points["right_cheek"][1] + 20],
                [chin[0], chin[1] + 10],
                [points["left_cheek"][0], points["left_cheek"][1] + 20]
            ], np.int32)

            mask_points = mask_points.reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [mask_points], color_bgr)

            # Draw mask bands
            ear_left = points["left_ear"]
            ear_right = points["right_ear"]

            cv2.line(overlay, (left_face[0], left_face[1]),
                     (ear_left[0] - 30, ear_left[1]), (70, 70, 70), 3)
            cv2.line(overlay, (right_face[0], right_face[1]),
                     (ear_right[0] + 30, ear_right[1]), (70, 70, 70), 3)

        elif style == "N95":
            # Get additional points for mask shape
            nose_tip = points["nose_tip"]
            left_cheek = points["left_cheek"]
            right_cheek = points["right_cheek"]
            chin = points["chin"]
            nose_bridge_top = points["nose_bridge_top"]
            nose_bridge_bottom = points["nose_bridge_bottom"]

            # Adjust mask width by scaling outward from center
            center_x = (left_cheek[0] + right_cheek[0]) // 2
            left_x = center_x - \
                int((center_x - left_cheek[0]) * mask_scale_factor)
            right_x = center_x + \
                int((right_cheek[0] - center_x) * mask_scale_factor)
            
            

            # Create polygon points for mask
            mask_points = np.array([
                [left_cheek[0], left_cheek[1]],
                [nose_bridge_top[0] - 15, nose_bridge_top[1]],
                [nose_bridge_bottom[0] + 15, nose_bridge_bottom[1]],
                [right_cheek[0], right_cheek[1]],
                [points["right_cheek"][0], points["right_cheek"][1] + 30],
                [chin[0], chin[1] + 15],
                [points["left_cheek"][0], points["left_cheek"][1] + 30]
            ], np.int32)

            mask_points = mask_points.reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [mask_points], color_bgr)

            # Draw mask edges with a different color
            cv2.polylines(overlay, [mask_points], True, (70, 70, 70), 2)

            # Draw a small valve circle
            valve_pos = (nose_tip[0], nose_tip[1] + 20)
            cv2.circle(overlay, valve_pos, 8, (70, 70, 70), -1)

            # Draw mask bands
            ear_left = points["left_ear"]
            ear_right = points["right_ear"]

            # Top band
            cv2.line(overlay, (left_cheek[0], left_cheek[1] - 10),
                     (ear_left[0] - 40, ear_left[1] - 30), (200, 200, 200), 3)
            cv2.line(overlay, (right_cheek[0], right_cheek[1] - 10),
                     (ear_right[0] + 40, ear_right[1] - 30), (200, 200, 200), 3)

            # Bottom band
            cv2.line(overlay, (left_cheek[0], left_cheek[1] + 15),
                     (ear_left[0] - 40, ear_left[1] + 20), (200, 200, 200), 3)
            cv2.line(overlay, (right_cheek[0], right_cheek[1] + 15),
                     (ear_right[0] + 40, ear_right[1] + 20), (200, 200, 200), 3)

        elif style == "Fashion":
            # Get mask dimensions
            face_contour_points = [points["left_cheek"],
                                   points["chin"], points["right_cheek"]]
            nose_bridge = [points["nose_bridge_top"],
                           points["nose_bridge_bottom"]]

            mask_height = int(
                abs(face_contour_points[1][1] - nose_bridge[0][1]) * 1.3)
            mask_width = int(
                abs(face_contour_points[0][0] - face_contour_points[2][0]) * 1.1)

            # Center point
            center_x = (face_contour_points[0]
                        [0] + face_contour_points[2][0]) // 2
            center_y = (face_contour_points[0][1] + nose_bridge[0][1]) // 2

            # Create polygon points for mask
            mask_points = np.array([
                [center_x - mask_width // 2, nose_bridge[0][1]],
                [center_x + mask_width // 2, nose_bridge[0][1]],
                [face_contour_points[2][0] + 10, face_contour_points[2][1] + 10],
                [face_contour_points[1][0], face_contour_points[1][1] + 15],
                [face_contour_points[0][0] - 10, face_contour_points[0][1] + 10]
            ], np.int32)

            mask_points = mask_points.reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [mask_points], color_bgr)

            # Add pattern (dots)
            for i in range(0, mask_width, 20):
                for j in range(0, mask_height, 20):
                    dot_x = center_x - mask_width // 2 + i
                    dot_y = nose_bridge[0][1] + j

                    # Check if the point is inside the mask
                    if cv2.pointPolygonTest(mask_points, (dot_x, dot_y), False) >= 0:
                        # Alternate color for pattern
                        pattern_color = (
                            255, 255, 255) if color != "White" else (0, 0, 0)
                        cv2.circle(overlay, (dot_x, dot_y),
                                   3, pattern_color, -1)

            # Add mask edges
            cv2.polylines(overlay, [mask_points], True, (70, 70, 70), 2)

            # Draw earloops
            ear_left = points["left_ear"]
            ear_right = points["right_ear"]

            cv2.line(overlay, (center_x - mask_width // 2, nose_bridge[0][1] + 10),
                     (ear_left[0] - 20, ear_left[1]), (70, 70, 70), 2)
            cv2.line(overlay, (center_x + mask_width // 2, nose_bridge[0][1] + 10),
                     (ear_right[0] + 20, ear_right[1]), (70, 70, 70), 2)
