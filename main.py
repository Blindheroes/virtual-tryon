#!/usr/bin/env python3
"""
Virtual Try-On System - Main Application
Main entry point for the Virtual Try-On application.
"""
import os
import sys
import tkinter as tk
from app.ui import VirtualTryOnApp
import cv2
import gc

def setup_assets_folder():
    """Create the assets folder structure if it doesn't exist."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(base_dir, "assets", "head_fashions")
    
    # Create main assets directory
    os.makedirs(assets_dir, exist_ok=True)
    
    # Create subdirectories for glasses
    glasses_dir = os.path.join(assets_dir, "glasses")
    os.makedirs(glasses_dir, exist_ok=True)
    for style in ["rectangle", "round", "aviator"]:
        os.makedirs(os.path.join(glasses_dir, style), exist_ok=True)
    
    # Create subdirectories for masks
    masks_dir = os.path.join(assets_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    for style in ["medical", "n95", "fashion"]:
        os.makedirs(os.path.join(masks_dir, style), exist_ok=True)
    
    return base_dir

def main():
    """Main application entry point."""
    # Setup assets folder
    base_dir = setup_assets_folder()
    
    # Configure OpenCV for better performance
    # Reduce resolution for faster processing
    cv2.setNumThreads(4)  # Limit OpenCV threads
    
    # Create and run the application
    root = tk.Tk()
    root.title("Virtual Try-On System")
    
    # Set minimum window size
    root.minsize(1000, 600)
    
    # Create the app
    app = VirtualTryOnApp(root, base_dir)
    
    # Set up proper closing
    def on_closing():
        app.on_closing()
        # Force garbage collection to clean up resources
        gc.collect()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Reduce tkinter update rate to improve performance
    def update_at_interval():
        root.update_idletasks()
        root.after(30, update_at_interval)  # ~30fps UI updates
    
    # Start the update loop
    update_at_interval()
    
    # Run the main loop
    root.mainloop()

if __name__ == "__main__":
    main()