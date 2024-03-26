import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Open the camera
cap = cv2.VideoCapture(0)

def classifyPose():
    # Function to calculate the slope
    def side_falldown(landmarks):
        x1, y1, z1 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z
        x2, y2, z2 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z
        shoulder_midpoint_x = (x1 + x2) / 2
        shoulder_midpoint_y = (y1 + y2) / 2
        shoulder_midpoint_z = (z1 + z2) / 2
        
        x3, y3, z3 = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z
        x4, y4, z4 = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z
        hip_midpoint_x = (x3 + x4) / 2
        hip_midpoint_y = (y3 + y4) / 2
        hip_midpoint_z = (z3 + z4) / 2
        
        if shoulder_midpoint_x - hip_midpoint_x == 0:
            return float('inf')
        else:
            slope = math.degrees(math.atan((hip_midpoint_y - shoulder_midpoint_y) / (hip_midpoint_x - shoulder_midpoint_x)))
            return slope

    # Function to calculate the knee angle
    def knee_angle(landmarks):
        x1, y1 = landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
        x2, y2 = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y
        x3, y3 = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y
        
        # Calculate vector a and b
        vec_a = np.array([x1 - x2, y1 - y2])
        vec_b = np.array([x3 - x2, y3 - y2])
        
        # Calculate dot product and magnitudes of vectors
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        # Calculate cosine similarity
        cos_theta = dot_product / (norm_a * norm_b)
        
        # Calculate angle (convert radians to degrees)
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        angle = np.degrees(angle)
        
        return angle

    # Initialize MediaPipe Pose model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Camera is disconnected.")
                continue
            
            # Image processing
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw pose landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                # Calculate slope using pose landmarks
                slope = side_falldown(results.pose_landmarks.landmark)
                print("Slope:", slope)
                
                # Calculate knee angle
                knee_angle_value = knee_angle(results.pose_landmarks.landmark)
                print("Knee Angle:", knee_angle_value)
                
                # Check if the slope is within a certain angle range and if the knee angle is below 90 degrees to detect falling
                if -65 <= abs(slope) <= 65 or -110 <= knee_angle_value <= 110:
                    print('Fall')
                    
            
            # Display the image
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break

    # Release the camera
    cap.release()

# Call the classifyPose function
classifyPose()

