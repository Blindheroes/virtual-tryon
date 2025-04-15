"""
Utility functions for the Virtual Try-On system.
"""
import os
import cv2
from PIL import Image, ImageTk
import numpy as np


def scan_assets_directory(base_dir):
    """
    Scan the assets directory and return available items.

    Args:
        base_dir: Base directory of the application

    Returns:
        Dictionary of available items
    """
    items = {
        "Glasses": {},
        "Mask": {}
    }

    # Path to assets
    assets_dir = os.path.join(base_dir, "assets", "head_fashions")

    # Check if directory exists
    if not os.path.exists(assets_dir):
        return items

    # Check for glasses
    glasses_dir = os.path.join(assets_dir, "glasses")
    if os.path.exists(glasses_dir):
        for style in os.listdir(glasses_dir):
            style_dir = os.path.join(glasses_dir, style)
            if os.path.isdir(style_dir):
                # Add style with capitalized name
                style_name = style.capitalize()
                items["Glasses"][style_name] = []

                # Scan for color variants
                for file in os.listdir(style_dir):
                    if file.lower().endswith(".png"):
                        color_name = os.path.splitext(file)[0].capitalize()
                        items["Glasses"][style_name].append(color_name)

    # Check for masks
    masks_dir = os.path.join(assets_dir, "masks")
    if os.path.exists(masks_dir):
        for style in os.listdir(masks_dir):
            style_dir = os.path.join(masks_dir, style)
            if os.path.isdir(style_dir):
                # Add style with capitalized name
                style_name = style.capitalize()
                items["Mask"][style_name] = []

                # Scan for color variants
                for file in os.listdir(style_dir):
                    if file.lower().endswith(".png"):
                        color_name = os.path.splitext(file)[0].capitalize()
                        items["Mask"][style_name].append(color_name)

    return items


def get_available_styles(items, item_type):
    """
    Get available styles for an item type.

    Args:
        items: Dictionary of items
        item_type: Type of item (Glasses or Mask)

    Returns:
        List of available styles
    """
    if item_type in items and items[item_type]:
        return list(items[item_type].keys())

    # Fallback default styles
    if item_type == "Glasses":
        return ["Rectangle", "Round", "Aviator"]
    else:  # Mask
        return ["Medical", "N95", "Fashion"]


def get_available_colors(items, item_type, style):
    """
    Get available colors for an item style.

    Args:
        items: Dictionary of items
        item_type: Type of item (Glasses or Mask)
        style: Style of the item

    Returns:
        List of available colors
    """
    if item_type in items and style in items[item_type] and items[item_type][style]:
        return items[item_type][style]

    # Fallback default colors
    return ["Black", "Blue", "Red", "Green", "White", "Yellow"]


def convert_cv_to_tk(frame):
    """
    Convert OpenCV frame to Tkinter compatible image.

    Args:
        frame: OpenCV frame (BGR)

    Returns:
        Tkinter compatible PhotoImage
    """
    try:
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        img = Image.fromarray(rgb_frame)

        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(image=img)

        return photo
    except Exception as e:
        print(f"Error converting image: {str(e)}")
        # Return a small blank image as fallback
        blank = Image.new('RGB', (10, 10), color='black')
        return ImageTk.PhotoImage(image=blank)


def generate_sample_png(output_dir, type_name, style_name, color_name, color_rgb):
    """
    Generate a sample PNG file for demonstration purposes.

    Args:
        output_dir: Output directory
        type_name: Type of item (glasses or masks)
        style_name: Style of the item
        color_name: Color name
        color_rgb: RGB color tuple

    Returns:
        Path to the generated file
    """
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create blank image with alpha channel (RGBA)
    if type_name == "glasses":
        width, height = 300, 100
    else:  # masks
        width, height = 300, 200

    # Create blank image with alpha
    img = np.zeros((height, width, 4), dtype=np.uint8)

    # Convert RGB to BGR for OpenCV
    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

    # Draw different shapes based on type and style
    if type_name == "glasses":
        if style_name == "rectangle":
            # Draw rectangular glasses
            cv2.rectangle(img, (50, 30), (120, 70), color_bgr, 2)
            cv2.rectangle(img, (180, 30), (250, 70), color_bgr, 2)
            cv2.line(img, (120, 50), (180, 50), color_bgr, 2)
            # Draw temples
            cv2.line(img, (50, 50), (20, 60), color_bgr, 2)
            cv2.line(img, (250, 50), (280, 60), color_bgr, 2)
            # Set alpha
            img[:, :, 3] = np.where(np.any(img[:, :, :3] > 0, axis=2), 255, 0)

        elif style_name == "round":
            # Draw round glasses
            cv2.circle(img, (85, 50), 35, color_bgr, 2)
            cv2.circle(img, (215, 50), 35, color_bgr, 2)
            cv2.line(img, (120, 50), (180, 50), color_bgr, 2)
            # Draw temples
            cv2.line(img, (50, 50), (20, 60), color_bgr, 2)
            cv2.line(img, (250, 50), (280, 60), color_bgr, 2)
            # Set alpha
            img[:, :, 3] = np.where(np.any(img[:, :, :3] > 0, axis=2), 255, 0)

        elif style_name == "aviator":
            # Draw aviator glasses
            pts_left = np.array([[65, 30], [105, 30], [120, 50], [
                                85, 70], [50, 50]], np.int32)
            pts_right = np.array([[195, 30], [235, 30], [250, 50], [
                                 215, 70], [180, 50]], np.int32)
            cv2.polylines(img, [pts_left], True, color_bgr, 2)
            cv2.polylines(img, [pts_right], True, color_bgr, 2)
            cv2.line(img, (120, 40), (180, 40), color_bgr, 2)
            # Draw temples
            cv2.line(img, (50, 40), (20, 60), color_bgr, 2)
            cv2.line(img, (250, 40), (280, 60), color_bgr, 2)
            # Set alpha
            img[:, :, 3] = np.where(np.any(img[:, :, :3] > 0, axis=2), 255, 0)

    else:  # masks
        if style_name == "medical":
            # Draw medical mask
            pts = np.array([[50, 50], [150, 30], [250, 50], [
                           250, 120], [150, 150], [50, 120]], np.int32)
            cv2.fillPoly(img, [pts], color_bgr)
            cv2.polylines(img, [pts], True, (100, 100, 100), 2)
            # Draw ear loops
            cv2.line(img, (50, 70), (10, 50), (100, 100, 100), 2)
            cv2.line(img, (250, 70), (290, 50), (100, 100, 100), 2)
            # Set alpha
            img[:, :, 3] = np.where(np.any(img[:, :, :3] > 0, axis=2), 255, 0)

        elif style_name == "n95":
            # Draw N95 mask
            pts = np.array([[50, 60], [150, 30], [250, 60], [
                           250, 130], [150, 160], [50, 130]], np.int32)
            cv2.fillPoly(img, [pts], color_bgr)
            cv2.polylines(img, [pts], True, (100, 100, 100), 2)
            # Draw valve
            cv2.circle(img, (150, 100), 15, (50, 50, 50), -1)
            # Draw straps
            cv2.line(img, (50, 70), (10, 40), (200, 200, 200), 3)
            cv2.line(img, (250, 70), (290, 40), (200, 200, 200), 3)
            cv2.line(img, (50, 120), (10, 150), (200, 200, 200), 3)
            cv2.line(img, (250, 120), (290, 150), (200, 200, 200), 3)
            # Set alpha
            img[:, :, 3] = np.where(np.any(img[:, :, :3] > 0, axis=2), 255, 0)

        elif style_name == "fashion":
            # Draw fashion mask
            pts = np.array([[50, 50], [150, 30], [250, 50], [
                           250, 120], [150, 150], [50, 120]], np.int32)
            cv2.fillPoly(img, [pts], color_bgr)

            # Add pattern
            for i in range(60, 240, 20):
                for j in range(50, 140, 20):
                    if cv2.pointPolygonTest(pts, (i, j), False) >= 0:
                        pattern_color = (255, 255, 255, 255) if sum(
                            color_rgb) < 500 else (0, 0, 0, 255)
                        cv2.circle(img, (i, j), 3, pattern_color, -1)

            # Add edges
            cv2.polylines(img, [pts], True, (100, 100, 100), 2)

            # Draw ear loops
            cv2.line(img, (50, 70), (10, 50), (100, 100, 100), 2)
            cv2.line(img, (250, 70), (290, 50), (100, 100, 100), 2)

            # Set alpha
            img[:, :, 3] = np.where(np.any(img[:, :, :3] > 0, axis=2), 255, 0)

    # Save image
    output_path = os.path.join(output_dir, f"{color_name.lower()}.png")
    cv2.imwrite(output_path, img)

    return output_path


def create_sample_assets(base_dir):
    """
    Create sample assets for demonstration.

    Args:
        base_dir: Base directory of the application
    """
    import numpy as np

    assets_dir = os.path.join(base_dir, "assets", "head_fashions")

    # Define colors
    colors = {
        "Red": (255, 0, 0),
        "Green": (0, 255, 0),
        "Blue": (0, 0, 255),
        "Black": (0, 0, 0),
        "White": (255, 255, 255),
        "Yellow": (255, 255, 0)
    }

    # Create sample glasses
    glasses_dir = os.path.join(assets_dir, "glasses")
    for style in ["rectangle", "round", "aviator"]:
        style_dir = os.path.join(glasses_dir, style)
        for color_name, color_rgb in colors.items():
            generate_sample_png(style_dir, "glasses",
                                style, color_name, color_rgb)

    # Create sample masks
    masks_dir = os.path.join(assets_dir, "masks")
    for style in ["medical", "n95", "fashion"]:
        style_dir = os.path.join(masks_dir, style)
        for color_name, color_rgb in colors.items():
            generate_sample_png(style_dir, "masks", style,
                                color_name, color_rgb)
