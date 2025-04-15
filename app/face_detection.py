"""
Face Detection Module
Handles face detection and landmark tracking for the Virtual Try-On system.
"""
import cv2
import mediapipe as mp
import numpy as np
import time
import copy

class FaceDetector:
    """Face detector class using MediaPipe Face Mesh."""
    
    def __init__(self):
        """Initialize the MediaPipe Face Mesh detector."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True  # Enable refined landmarks for better stability
        )
        
        # Store previous landmarks for smoothing
        self.prev_landmarks = None
        self.smoothing_factor = 0.3  # Lower = more smoothing
        self.last_detection_time = 0
        
    def detect_face(self, frame):
        """
        Detect face landmarks in the given frame.
        
        Args:
            frame: BGR frame from the camera
            
        Returns:
            Face landmarks if detected, None otherwise
        """
        # Rate limiting for performance
        current_time = time.time()
        if current_time - self.last_detection_time < 0.02:  # Max ~50 fps for detection
            return self.prev_landmarks
            
        self.last_detection_time = current_time
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image with MediaPipe Face Mesh
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            # Get the landmarks
            current_landmarks = results.multi_face_landmarks[0].landmark
            
            # Apply temporal smoothing if we have previous landmarks
            if self.prev_landmarks is not None:
                smoothed_landmarks = []
                for i, landmark in enumerate(current_landmarks):
                    # Create a smoothed landmark by copying the current one
                    smoothed_landmark = copy.deepcopy(landmark)
                    
                    # Apply exponential smoothing
                    smoothed_landmark.x = self.smoothing_factor * landmark.x + (1 - self.smoothing_factor) * self.prev_landmarks[i].x
                    smoothed_landmark.y = self.smoothing_factor * landmark.y + (1 - self.smoothing_factor) * self.prev_landmarks[i].y
                    smoothed_landmark.z = self.smoothing_factor * landmark.z + (1 - self.smoothing_factor) * self.prev_landmarks[i].z
                    
                    smoothed_landmarks.append(smoothed_landmark)
                
                self.prev_landmarks = smoothed_landmarks
                return smoothed_landmarks
            else:
                # First detection, store landmarks as-is
                self.prev_landmarks = list(current_landmarks)
                return current_landmarks
        
        # No face detected
        return None
    
    def get_face_dimensions(self, landmarks, img_width, img_height):
        """
        Calculate face dimensions based on landmarks.
        
        Args:
            landmarks: Face landmarks from MediaPipe
            img_width: Image width
            img_height: Image height
            
        Returns:
            Dictionary containing face dimensions and key points
        """
        if landmarks is None:
            return None
        
        try:
            # Convert key points to pixel coordinates
            points = {}
            
            # Eyes
            points["left_eye"] = (int(landmarks[33].x * img_width), int(landmarks[33].y * img_height))
            points["right_eye"] = (int(landmarks[263].x * img_width), int(landmarks[263].y * img_height))
            
            # Nose
            points["nose_tip"] = (int(landmarks[4].x * img_width), int(landmarks[4].y * img_height))
            
            # Face contour
            points["left_cheek"] = (int(landmarks[152].x * img_width), int(landmarks[152].y * img_height))
            points["chin"] = (int(landmarks[10].x * img_width), int(landmarks[10].y * img_height))
            points["right_cheek"] = (int(landmarks[378].x * img_width), int(landmarks[378].y * img_height))
            
            # Temples
            points["left_temple"] = (int(landmarks[162].x * img_width), int(landmarks[162].y * img_height))
            points["right_temple"] = (int(landmarks[389].x * img_width), int(landmarks[389].y * img_height))
            
            # Ears
            points["left_ear"] = (int(landmarks[234].x * img_width), int(landmarks[234].y * img_height))
            points["right_ear"] = (int(landmarks[454].x * img_width), int(landmarks[454].y * img_height))
            
            # Nose bridge
            points["nose_bridge_top"] = (int(landmarks[6].x * img_width), int(landmarks[6].y * img_height))
            points["nose_bridge_bottom"] = (int(landmarks[197].x * img_width), int(landmarks[197].y * img_height))
            
            # Calculate face dimensions
            dimensions = {}
            dimensions["eye_center"] = ((points["left_eye"][0] + points["right_eye"][0]) // 2,
                                    (points["left_eye"][1] + points["right_eye"][1]) // 2)
            
            dimensions["face_width"] = abs(points["right_cheek"][0] - points["left_cheek"][0])
            dimensions["face_height"] = abs(points["chin"][1] - dimensions["eye_center"][1])
            
            # Compile results
            result = {
                "points": points,
                "dimensions": dimensions
            }
            
            return result
            
        except Exception as e:
            print(f"Error in get_face_dimensions: {str(e)}")
            return None
    
    def release(self):
        """Release resources."""
        self.prev_landmarks = None