import numpy as np
import cv2
import mediapipe as mp
from datetime import datetime
import plotly.graph_objects as go
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from torchvision import transforms
import os
import time
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

FACE_CAMERA_POSITION = np.array([0, 0, 0])  # Face camera position in 3D space (in cm)
DRIVER_POSITION = np.array([0,10, -70]) # Driver position in 3D space (in cm)
GAZE_CAMERA_POSITION = np.array([-30, 0, -80]) # Gaze camera position in 3D space (in cm)
DIAGONAL_FOV = 70  # Diagonal FoV for Akaso EK7000Pro (degrees)
SCREEN_DISTANCE = 70  # Distance of the screen from the driver (in cm)
GAZE_CAMERA_FOCAL_LENGTH = 650  # Focal length of the gaze camera (in pixels)

class SingleInputModel(nn.Module):
    def __init__(self, num_classes, freeze_base=False):
        super(SingleInputModel, self).__init__()
        self.backbone = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()  # Remove original classification head

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256, momentum=0.1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

        # Freeze early layers
        if freeze_base:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        features = self.backbone(x)
        out = self.fc(features)
        return out
    
class DualInputModel(nn.Module):
    def __init__(self, num_classes, freeze_base=False):
        super(DualInputModel, self).__init__()
        self.shared_cnn = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        in_features = self.shared_cnn.fc.in_features
        self.shared_cnn.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(in_features * 2, 256),
            nn.BatchNorm1d(256, momentum=0.1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

        # Freeze early layers
        if freeze_base:
            for name, param in self.shared_cnn.named_parameters():
                param.requires_grad = False

    def forward(self, x):
        face_input, hand_input = x
        face_input = face_input.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        hand_input = hand_input.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        face_features = self.shared_cnn(face_input)
        hand_features = self.shared_cnn(hand_input)

        combined_features = torch.cat((face_features, hand_features), dim=1)
        out = self.fc(combined_features)
        return out
    
class SingleInputCombinedModel(nn.Module):
    def __init__(self, num_classes, freeze_base=False):
        super(SingleInputModel, self).__init__()
        self.backbone = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()  # Remove original classification head
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256, momentum=0.1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

        # Freeze early layers
        if freeze_base:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        features = self.backbone(x)
        out = self.fc(features)
        return out

class ActionClassifier:
    def __init__(self, model, model_path, model_inputs=1):
        self.model = model
        self.model.load_state_dict(torch.load(model_path,weights_only=True, map_location='cuda'))
        self.model = self.model.to('cuda').eval()
        self.model_inputs = model_inputs
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
        ])

        self.classes = [
            'change_gear', 'drinking', 'hair_and_makeup', 'phonecall_left', 'phonecall_right',
            'radio', 'reach_backseat', 'reach_side', 'safe_drive', 'standstill_or_waiting',
            'talking_to_passenger', 'texting_left', 'texting_right', 'unclassified'
        ]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def preprocess_video(self, video_frame, sequence_length=16):
        """Load and preprocess video into a tensor."""
        frames = []

        while len(frames) < sequence_length:
            frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
            frame = self.transform(frame)  # Apply the transformations
            frames.append(frame)

        # Pad or truncate to match the sequence length
        while len(frames) < sequence_length:
            frames.append(frames[-1])  # Duplicate the last frame if video is too short
        
        video_tensor = torch.stack(frames[:sequence_length])  # [T, C, H, W]
        return video_tensor
        
    def classify_action_dual_inputs(self, video_tensor):
        with torch.no_grad():
            output = self.model(video_tensor)
            _, predicted = torch.max(output, 1)
            return self.classes[predicted.item()]
        
    def classify_action_single_input(self, face_tensor):
        with torch.no_grad():
            output = self.model(face_tensor)
            _, predicted = torch.max(output, 1)
            return self.classes[predicted.item()]

    def classify_action_single_input_combine(self, face_tensor, gaze_tensor):
        with torch.no_grad():
            face_output = self.model(face_tensor)
            gaze_output = self.model(gaze_tensor)
            face_output = nn.functional.softmax(face_output, dim=1)
            gaze_output = nn.functional.softmax(gaze_output, dim=1)
            output = (face_output + gaze_output) / 2
            _, predicted = torch.max(output, 1)
            return self.classes[predicted.item()]   
              
    def process_frame(self, face_video, gaze_video=None):
        if gaze_video is not None:
            face_tensor = self.preprocess_video(face_video).unsqueeze(0).to(self.device)
            gaze_tensor = self.preprocess_video(gaze_video).unsqueeze(0).to(self.device)
            if self.model_inputs == 1:
                action = self.classify_action_single_input_combine(face_tensor, gaze_tensor)
                return action
            else:
                video_tensor = (face_tensor, gaze_tensor)
                action = self.classify_action_dual_inputs(video_tensor)
                return action
        else:
            face_tensor = self.preprocess_video(face_video).unsqueeze(0).to(self.device)  # Add batch dimension
            action = self.classify_action_single_input(face_tensor)
            return action
             
class GazeEstimator:
    def __init__(self, driver_position, gaze_camera_position, screen_distance, gaze_camera_focal_length, gaze_camera_diagonal_fov):
        self.driver_position = driver_position
        self.gaze_camera_position = gaze_camera_position
        self.screen_distance = screen_distance
        self.gaze_camera_focal_length = gaze_camera_focal_length
        self.gaze_camera_diagonal_fov = gaze_camera_diagonal_fov
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True
        )

        # Initialize default ROIs
        self.rois = {
            "windscreen": {"coords": [(46, 184), (546, 331)], "severity": 0.1, "color": (0, 255, 0)},   # Green
            "dashboard": {"coords": [(46, 345), (546, 478)], "severity": 0.7,"color": (0, 0, 255)},  # Red
            "left_mirror": {"coords": [(0, 183), (45, 478)], "severity": 0.3, "color": (255, 0, 0)},  # Blue
            "right_mirror": {"coords": [(547, 184), (639, 478)], "severity": 0.3, "color": (255, 0, 0)},  # Blue
        }

    @staticmethod
    def _calculate_head_pose(landmarks):
        # Yaw (Rotation around Y-axis)
        R_y = (landmarks[362].z - landmarks[133].z) / (landmarks[362].x - landmarks[133].x)

        # Pitch (Rotation around X-axis)
        R_x = (landmarks[168].z - landmarks[2].z) / (landmarks[168].y - landmarks[2].y)

        # Roll (Rotation around Z-axis) – Eye alignment
        dx = landmarks[362].x - landmarks[133].x
        dy = landmarks[362].y - landmarks[133].y
        R_z = np.degrees(np.arctan2(dy, dx))  # Roll calculated from eye corner difference

        return np.degrees(R_y), np.degrees(R_x), R_z

    @staticmethod
    def estimate_gaze_from_iris(iris_center, eye_center):
        displacement = np.array(iris_center) - np.array(eye_center)

        # Normalize the displacement
        norm_displacement = displacement / np.linalg.norm(displacement)

        # Calculate yaw and pitch in degrees
        yaw = np.degrees(np.arctan(norm_displacement[0]))  # Horizontal angle
        pitch = np.degrees(np.arctan(norm_displacement[1]))  # Vertical angle

        return yaw, pitch
    
    @staticmethod
    def get_eye_landmarks(landmarks):
        # Right eye landmarks (Anatomical Right)
        right_iris = (landmarks[468].x, landmarks[468].y)
        right_outer = (landmarks[33].x, landmarks[33].y)
        right_inner = (landmarks[133].x, landmarks[133].y)
        right_eye_center = (
            (right_outer[0] + right_inner[0]) / 2,
            (right_outer[1] + right_inner[1]) / 2
        )

        # Left eye landmarks (Anatomical Left)
        left_iris = (landmarks[473].x, landmarks[473].y)
        left_outer = (landmarks[263].x, landmarks[263].y)
        left_inner = (landmarks[362].x, landmarks[362].y)
        left_eye_center = (
            (left_outer[0] + left_inner[0]) / 2,
            (left_outer[1] + left_inner[1]) / 2
        )
        
        return right_iris, right_eye_center, left_iris, left_eye_center

    @staticmethod
    def combine_head_eye_angles(R_y, R_x, left_yaw, left_pitch, right_yaw, right_pitch, w_head=0.8, w_eye=0.2):
        """
        Combines head and eye angles to calculate the unified gaze direction using weighted combination.
        
        Parameters:
        - R_y, R_x: Head yaw and pitch (in degrees).
        - left_yaw, left_pitch: Left eye yaw and pitch (in degrees).
        - right_yaw, right_pitch: Right eye yaw and pitch (in degrees).
        - w_head: Weight for the head angles.
        - w_eye: Weight for the eye angles.
        
        Returns:
        - combined_yaw: Unified yaw angle.
        - combined_pitch: Unified pitch angle.
        """
        # Unify eye angles using the average of left and right eye angles
        unified_eye_yaw = (left_yaw + right_yaw) / 2
        unified_eye_pitch = (left_pitch + right_pitch) / 2

        # Combine head and eye angles with weights
        combined_yaw = round(w_head * R_y + w_eye * unified_eye_yaw, 2)
        combined_pitch = round(w_head * R_x + w_eye * unified_eye_pitch, 2)

        return combined_yaw, combined_pitch

    @staticmethod
    def euler_to_gaze_vector(yaw, pitch):
        # Convert degrees to radians
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(-pitch)
        
        # Calculate gaze direction vector
        g_x = np.cos(pitch_rad) * np.sin(yaw_rad)
        g_y = np.sin(pitch_rad)
        g_z = np.cos(pitch_rad) * np.cos(yaw_rad)
        return np.array([g_x, g_y, g_z])
    
    @staticmethod
    def calculate_pog(eye_position, gaze_direction, screen_distance=0):
        # Normalize the gaze direction vector
        gaze_direction = gaze_direction / np.linalg.norm(gaze_direction)
        
        # Calculate the intersection point with the screen plane (z = 0)
        t = (screen_distance - eye_position[2]) / gaze_direction[2]
        intersection = eye_position + t * gaze_direction
        
        return intersection

    @staticmethod
    def calculate_gaze_direction(observer_position, intersection):
        # Calculate gaze direction vector from observer to intersection point
        gaze_direction = intersection - observer_position
        
        # Normalize the gaze direction vector
        gaze_direction = gaze_direction / np.linalg.norm(gaze_direction)
        
        return gaze_direction

    @staticmethod
    def calculate_fov_components(diagonal_fov, aspect_ratio=(4, 3)):
        aspect_width, aspect_height = aspect_ratio
        diag_rad = np.radians(diagonal_fov / 2)
        
        # Horizontal and Vertical FoV calculation
        fov_x = 2 * np.degrees(np.arctan(np.tan(diag_rad) * (aspect_width / np.sqrt(aspect_width**2 + aspect_height**2))))
        fov_y = 2 * np.degrees(np.arctan(np.tan(diag_rad) * (aspect_height / np.sqrt(aspect_width**2 + aspect_height**2))))
        
        return fov_x, fov_y
    
    @staticmethod
    def calculate_fov(frame, focal_length):
        frame_width, frame_height = frame.shape[1], frame.shape[0]
        fov_x = 2 * np.degrees(np.arctan(frame_width / (2 * focal_length)))
        fov_y = 2 * np.degrees(np.arctan(frame_height / (2 * focal_length)))
        return fov_x, fov_y

    @staticmethod
    def is_pog_visible(point_of_gaze, focal_length, fov_x, fov_y, screen_distance):
        # Calculate visible bounds
        half_width = screen_distance * np.tan(np.radians(fov_x / 2))
        half_height = screen_distance * np.tan(np.radians(fov_y / 2))
        
        # Check if the PoG lies within the bounds
        x, y, z = point_of_gaze
        if -half_width <= x <= half_width and -half_height <= y <= half_height:
            return True  # PoG is within bounds
        return False  # PoG is out of bounds

    @staticmethod
    def scale_pog_to_screen(observer_pog, sensor_width, sensor_height, screen_distance, fov_x, fov_y):
        # Scale x and y based on the observer's FoV
        half_width = screen_distance * np.tan(np.radians(fov_x / 2))
        half_height = screen_distance * np.tan(np.radians(fov_y / 2))
        
        # Normalize observer_pog
        normalized_x = (observer_pog[0] + half_width) / (2 * half_width)
        normalized_y = (half_height - observer_pog[1]) / (2 * half_height)
        
        # Map to screen resolution
        pixel_x = int(normalized_x * sensor_width)
        pixel_y = int(normalized_y * sensor_height)
        return pixel_x, pixel_y
    
    def get_gaze_zone(self, observer_pog, gaze_frame, fov_x, fov_y):
        gaze_frame_width, gaze_frame_height, _ = gaze_frame.shape
        # Scale observer_pog to pixel coordinates
        pixel_x, pixel_y = self.scale_pog_to_screen(observer_pog, gaze_frame_width, gaze_frame_height, self.screen_distance, fov_x, fov_y)

        for zone, details in self.rois.items():
            top_left, bottom_right = details["coords"]
            if top_left[0] <= pixel_x <= bottom_right[0] and top_left[1] <= pixel_y <= bottom_right[1]:
                return zone, details["severity"]
        return "out_of_bounds", 1.0  # Default high severity for out-of-bounds gaze

    
    @staticmethod
    def calculate_EAR(landmarks, eye_indices):
        p1 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
        p2 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
        p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
        p4 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
        p5 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
        p6 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
        
        # Vertical distances
        vertical_1 = np.linalg.norm(p2 - p4)
        vertical_2 = np.linalg.norm(p3 - p5)
        # Horizontal distance
        horizontal = np.linalg.norm(p1 - p6)
        
        # EAR formula
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def _process_face_landmarks(self, driver_frame, observer_frame, landmarks):
        # Estimate gaze angles and unify (limit to 2 decimal places)
        R_y, R_x, R_z = map(lambda x: round(x, 2), self._calculate_head_pose(landmarks))
        left_iris, left_eye_center, right_iris, right_eye_center = self.get_eye_landmarks(
                            landmarks
                        )
        right_eye_ear = self.calculate_EAR(landmarks, [33, 160, 158, 133, 153, 144])
        left_eye_ear = self.calculate_EAR(landmarks, [263, 387, 385, 362, 380, 373])
        ear = (right_eye_ear + left_eye_ear) / 2

        
        left_yaw, left_pitch = map(lambda x: round(x, 2), self.estimate_gaze_from_iris(left_iris, left_eye_center))
        right_yaw, right_pitch = map(lambda x: round(x, 2), self.estimate_gaze_from_iris(right_iris, right_eye_center))
        unified_yaw, unified_pitch = self.combine_head_eye_angles(R_y, R_x, left_yaw, left_pitch, right_yaw, right_pitch)
        
        # Calculate observer PoG and gaze zone
        eye_gaze_direction = self.euler_to_gaze_vector(unified_yaw, unified_pitch)
        eye_pog = self.calculate_pog(self.driver_position, eye_gaze_direction, self.screen_distance)
        observer_gaze_direction = self.calculate_gaze_direction(self.gaze_camera_position, eye_pog)
        observer_pog = self.calculate_pog(self.gaze_camera_position, observer_gaze_direction, np.abs(self.gaze_camera_position[2]))
        fov_x, fov_y = self.calculate_fov(observer_frame, self.gaze_camera_focal_length)
        gaze_zone, gaze_score = self.get_gaze_zone(observer_pog, observer_frame, fov_x, fov_y)
        is_visible = self.is_pog_visible(observer_pog, self.gaze_camera_focal_length, fov_x, fov_y, self.screen_distance)

        return (R_y, R_x, R_z), (unified_yaw, unified_pitch), eye_gaze_direction, eye_pog, observer_gaze_direction, observer_pog, (fov_x, fov_y), is_visible, ear , (gaze_zone, gaze_score)

    def draw_gaze_visualization(self, face_frame, gaze_frame, observer_pog,fov ,euler_angles = None):
        
        # Draw the observer point of gaze on the gaze frame
        gaze_frame_height, gaze_frame_width, _ = gaze_frame.shape
        fov_x, fov_y = fov
        # Scale observer_pog to pixel coordinates
        pixel_x, pixel_y = self.scale_pog_to_screen(observer_pog, gaze_frame_width, gaze_frame_height, self.screen_distance, fov_x, fov_y)

        # Ensure pixel coordinates are within screen bounds
        pixel_x = np.clip(pixel_x, 0, gaze_frame_width - 1)
        pixel_y = np.clip(pixel_y, 0, gaze_frame_height - 1)

        # Draw PoG on the screen
        cv2.circle(gaze_frame, (pixel_x, pixel_y), 10, (0, 255, 0), -1)
        cv2.putText(gaze_frame, f"PoG: ({pixel_x}, {pixel_y})", (pixel_x + 10, pixel_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw the unified yaw and pitch arrow on the face frame
        face_frame_height, face_frame_width, _ = face_frame.shape
        center_x, center_y = face_frame_width // 2, face_frame_height // 2
        
        if euler_angles is not None:
            arrow_length = 100
            unified_yaw, unified_pitch = euler_angles
            end_x = int(center_x + arrow_length * np.sin(np.radians(unified_yaw)))
            end_y = int(center_y + arrow_length * np.sin(np.radians(-unified_pitch)))
            cv2.arrowedLine(face_frame, (center_x, center_y), (end_x, end_y), (255, 0, 0), 2)
            cv2.putText(face_frame, f"Yaw: {unified_yaw:.2f}, Pitch: {unified_pitch:.2f}", (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return face_frame, gaze_frame

    
    def process_frame(self, face_frame, gaze_frame):
        # Initialize default values
        processed_face_frame, processed_gaze_frame = face_frame, gaze_frame
        head_euler = gaze_euler = eye_gaze_direction = eye_pog = observer_gaze_direction = observer_pog = fov = is_visible = ear = zone_detail = None

        for roi, details in self.rois.items():
            top_left, bottom_right = details["coords"]
            color = details["color"]
            cv2.rectangle(processed_gaze_frame, top_left, bottom_right, color, 2)

        results = self.face_mesh.process(face_frame)
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                # Draw the face mesh on the face_frame
                for idx, landmark in enumerate(landmarks.landmark):
                    h, w, _ = face_frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(face_frame, (x, y), 1, (0, 255, 0), -1)

                head_euler, gaze_euler, eye_gaze_direction, eye_pog, observer_gaze_direction, observer_pog, fov, is_visible, ear, zone_detail = self._process_face_landmarks(face_frame, gaze_frame, landmarks.landmark)
                if is_visible:
                    cv2.putText(face_frame, "PoG Visible", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    processed_face_frame, processed_gaze_frame = self.draw_gaze_visualization(face_frame, gaze_frame, observer_pog, fov, gaze_euler,)
                else:
                    cv2.putText(face_frame, "PoG Not Visible", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    processed_face_frame, processed_gaze_frame = face_frame, gaze_frame

        return processed_face_frame, processed_gaze_frame, (head_euler, gaze_euler, eye_gaze_direction, eye_pog, observer_gaze_direction, observer_pog, fov, is_visible, ear, zone_detail)

class DriverDistractionDetector:
    def __init__(self, ear_weight=0.3, pog_weight=0.2, action_weight=0.5, 
                 ear_threshold=0.2, distraction_threshold=0.5):
        self.action_severity = {
            "change_gear": 0.2,
            "drinking": 0.4,
            "hair_and_makeup": 0.9,
            "phonecall_left": 0.7,
            "phonecall_right": 0.7,
            "radio": 0.5,
            "reach_backseat": 0.8,
            "reach_side": 0.6,
            "safe_drive": 0.0,
            "standstill_or_waiting": 0.1,
            "talking_to_passenger": 0.5,
            "texting_left": 0.9,
            "texting_right": 0.9,
            "unclassified": 0.3,
        }
        self.ear_weight = ear_weight
        self.pog_weight = pog_weight
        self.action_weight = action_weight # Action severity weight
        self.ear_threshold = ear_threshold # EAR threshold
        self.distraction_threshold = distraction_threshold # Distraction threshold

    def get_action_severity(self, detected_action):
        """
        Get the severity weight of the detected action.
        """
        return self.action_severity.get(detected_action, 0)

    def detect_distraction(self, detected_action, gaze_details):
        """
        Detect distraction based on action severity, EAR, and time off ROI.
        """
        if any(detail is None for detail in gaze_details):
            return False, 0.0, (0.0, 0.0, 0.0), "Not Valid"
        else:
            head_euler, gaze_euler, eye_gaze_direction, eye_pog, observer_gaze_direction, observer_pog, fov, is_visible, ear, zone_details = gaze_details
            gaze_zone, gaze_severity = zone_details

            # Calculate individual scores
            action_score = self.action_weight * self.get_action_severity(detected_action)
            ear_score = self.ear_weight * (1 if ear < self.ear_threshold else 0)
            pog_score = self.pog_weight * gaze_severity

            # Normalize and calculate total score
            total_score = action_score + ear_score + pog_score
            total_score = min(total_score, 1.0)  # Cap score at 1.0

            # Determine distraction
            is_distracted = total_score >= self.distraction_threshold
            return is_distracted, total_score, (action_score, ear_score, pog_score), "Valid"


def annotate_results(frame, action_class, gaze_details, is_distracted, pipeline_fps):  
    """
    Annotate the combined results on a blank frame.
    """
    head_euler, gaze_euler, eye_gaze_direction, eye_pog, observer_gaze_direction, observer_pog, fov, is_visible, ear, zone_details = gaze_details
    distraction, total_score, other_scores, score_status = is_distracted
    action_score, ear_score, pog_score = other_scores

    # Annotate pipeline time
    cv2.putText(frame, f"Pipeline FPS: {pipeline_fps:.2f} s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # Annotate action
    cv2.putText(frame, f"Action: {action_class}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if eye_pog is not None and observer_pog is not None:
        cv2.putText(frame, f"Driver PoG: {np.round(eye_pog, 2)}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Observer PoG: {np.round(observer_pog,2)}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Annotate gaze zone and score
    if is_visible:
        gaze_zone, gaze_score = zone_details
        cv2.putText(frame, f"Gaze Zone: {gaze_zone}", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Gaze Score: {gaze_score}", (650, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Gaze Zone: Undetermined", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Annotate EAR
    if ear is not None:
        EAR_status = "Open" if ear > 0.2 else "Closed"
        cv2.putText(frame, f"EAR: {ear:.2f} || STATUS: {EAR_status}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    else:
        cv2.putText(frame, f"EAR: {None} || STATUS: Not Detected", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if score_status == "Valid":
        # Annotate distraction status
        distraction_status = "Distracted" if distraction else "Not Distracted"
        color = (0, 0, 255) if distraction else (0, 255, 0)
        cv2.putText(frame, f"Distraction: {distraction_status}", (650, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f"Action Score: {action_score:.2f}", (650, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f"EAR Score: {ear_score:.2f}", (650, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f"PoG Score: {pog_score:.2f}", (650, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f"Distraction Score: {total_score:.2f}", (650, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        cv2.putText(frame, "Distraction: Invalid Score", (650, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame


def process_pipeline(face_frame, gaze_frame):
    """
    Central processing pipeline for driver action, gaze, and distraction detection.
    """
    # Step 1: Preprocess frames
    face_frame = cv2.flip(face_frame, 1) # for webcam

    blank_frame = np.zeros((240,1280,3), dtype=np.uint8)
    start_time = time.time()

    # Step 2: Driver Action Classification
    action_class = driver_actions_classifier.process_frame(face_frame)
    # print(action_class)

    # Step 3: Gaze Estimation
    processed_face, processed_gaze, gaze_details = gaze_estimator.process_frame(face_frame, gaze_frame)

    # Step 4: Distraction Detection
    is_distracted = driver_distraction_detector.detect_distraction(action_class, gaze_details)
    
    end_time = time.time()
    pipline_time = end_time - start_time
    pipeline_fps = 1 / pipline_time
    # Step 5: Annotation
    annotated_frame = annotate_results(blank_frame, action_class, gaze_details, is_distracted, pipeline_fps)

    return processed_face, processed_gaze, annotated_frame, is_distracted


if __name__ == "__main__":
    
    driver_actions_classifier = ActionClassifier(
        model=SingleInputModel(14), model_path="./DriverActionDetectionSystem/DMD Dataset/driver_actions_best_model_r2plus1d_singe_input_face_final.pth", model_inputs=1
    )

    gaze_estimator = GazeEstimator(
        driver_position=DRIVER_POSITION,
        gaze_camera_position=GAZE_CAMERA_POSITION,
        screen_distance=SCREEN_DISTANCE,
        gaze_camera_focal_length=GAZE_CAMERA_FOCAL_LENGTH,
        gaze_camera_diagonal_fov=DIAGONAL_FOV,
    )

    driver_distraction_detector = DriverDistractionDetector(
        ear_weight=0.3,
        pog_weight=0.2,
        ear_threshold=0.2,
    )

    # face_camera = cv2.VideoCapture("./test_video/FaceCamera/camera1_output.avi")
    # gaze_camera = cv2.VideoCapture("./test_video/GazeCamera/camera2_output.avi")
    face_camera = cv2.VideoCapture(0)
    gaze_camera = cv2.VideoCapture(1)

  # Initialize VideoWriter
    forcc = cv2.VideoWriter_fourcc(*'XVID')
    video_filename = f"gaze_result_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.avi"
    out = cv2.VideoWriter(video_filename, forcc, 20.0, (1280, 720))

    if not out.isOpened():
        print("Error: Unable to initialize VideoWriter.")
        exit(1)

    # Main loop
    while face_camera.isOpened() and gaze_camera.isOpened():
        face_ret, face_frame = face_camera.read()
        gaze_ret, gaze_frame = gaze_camera.read()

        if not face_ret or not gaze_ret:
            break
        
        # Centralized pipeline
        processed_face, processed_gaze, annotated_frame, is_distracted = process_pipeline(face_frame, gaze_frame)
        
        # Combine frames for display
        # Horizontally stack processed_face and processed_gaze
        combined_frame = np.hstack((processed_face, processed_gaze))

        # Resize annotated_frame to match the width of combined_frame
        annotated_frame_resized = cv2.resize(annotated_frame, (combined_frame.shape[1], annotated_frame.shape[0]))

        # Vertically stack combined_frame and annotated_frame
        final_frame = np.vstack((combined_frame, annotated_frame_resized))

        # Display frames
        cv2.imshow("Driver Monitoring System", final_frame)
        out.write(final_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break


    out.release()
    face_camera.release()
    gaze_camera.release()
    cv2.destroyAllWindows()
    print(f"Video saved as {video_filename}")

    