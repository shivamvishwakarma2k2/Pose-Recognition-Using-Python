# Importing OpenCV, MediaPipe, NumPy and Matplotlib Libraray
import cv2 
import mediapipe as mp  
import numpy as np  
import matplotlib.pyplot as plt  

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose  # Access the pose solution from MediaPipe
mp_draw = mp.solutions.drawing_utils  # Access drawing utilities for visualizing landmarks
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)  # Initialize pose detection with specific settings

"""Calculate angle between three points."""
def calculate_angle(a, b, c):
    # Convert points to numpy arrays for vector calculations
    try:
        a = np.array([a.x, a.y])  # Convert point a to a NumPy array
        b = np.array([b.x, b.y])  # Convert point b to a NumPy array
        c = np.array([c.x, c.y])  # Convert point c to a NumPy array
        
        # Calculate vectors and cosine of the angle
        ab = a - b  # Vector from b to a
        bc = c - b  # Vector from b to c
        
        # Calculate the cosine of the angle using the dot product
        cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Ensure the value is within the valid range for arccos
        return np.degrees(angle)  # Convert the angle from radians to degrees
    except:
        return 0  # Return 0 if an error occurs

def calculate_body_angles(landmarks):
    """Calculate all relevant body angles for pose detection."""
    angles = {}  # Dictionary to store calculated angles
    
    try:
        # Calculate neck angles using landmarks
        angles['neck_angle_left'] = calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_EAR],
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
            landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        )
        angles['neck_angle_right'] = calculate_angle(
            landmarks[mp_pose.PoseLandmark.RIGHT_EAR],
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        )
        
        # Calculate back angles using landmarks
        angles['back_angle_left'] = calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
            landmarks[mp_pose.PoseLandmark.LEFT_HIP],
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        )
        angles['back_angle_right'] = calculate_angle(
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        )
        
        # Calculate knee angles using landmarks
        angles['knee_angle_left'] = calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_HIP],
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        )
        angles['knee_angle_right'] = calculate_angle(
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE],
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
        )
        
        # Calculate arm angles using landmarks
        angles['arm_angle_left'] = calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        )
        angles['arm_angle_right'] = calculate_angle(
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        )
        
        # Calculate head tilt using landmarks
        angles['head_tilt'] = calculate_angle(
            landmarks[mp_pose.PoseLandmark.RIGHT_EAR],
            landmarks[mp_pose.PoseLandmark.NOSE],
            landmarks[mp_pose.PoseLandmark.LEFT_EAR]
        )
        
        return angles  # Return the dictionary of angles
    except Exception as e:
        print(f"Error calculating angles: {str(e)}")  # Print error message if an exception occurs
        return None  # Return None if an error occurs

def check_standing_posture(angles, landmarks):
    """Analyze standing posture and return specific standing pose."""
    # Check if knee angles indicate a standing posture
    if (160 <= angles['knee_angle_left'] <= 180 and 
        160 <= angles['knee_angle_right'] <= 180):
        return "Standing"  # Return "Standing" if conditions are met
    return None  # Return None if not standing

def check_sitting_posture(angles, landmarks):
    """Analyze sitting posture and return specific sitting pose."""
    # Check if knee angles indicate a sitting posture
    if (85 <= angles['knee_angle_left'] <= 135 and 
        85 <= angles['knee_angle_right'] <= 135):
        
        # Check neck angles to determine slouching
        if angles['neck_angle_left'] < 150 or angles['neck_angle_right'] < 150:
            return "Slouching while Sitting"  # Return slouching status
        return "Sitting with Good Posture"  # Return good posture status
    return None  # Return None if not sitting

def classify_pose(landmarks):
    """Main pose classification function."""
    # Check if landmarks are detected
    if not landmarks:
        return "No Person Detected"  # Return if no landmarks are found

    # Calculate all body angles
    angles = calculate_body_angles(landmarks)
    if not angles:
        return "Error Calculating Pose"  # Return error if angles could not be calculated

    # Check basic postures first
    standing_pose = check_standing_posture(angles, landmarks)
    if standing_pose:
        return standing_pose  # Return standing pose if detected

    sitting_pose = check_sitting_posture(angles, landmarks)
    if sitting_pose:
        return sitting_pose  # Return sitting pose if detected

    # Check for sleeping based on shoulder-hip vertical difference
    shoulder_hip_vertical_diff = abs(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y - 
                                   landmarks[mp_pose.PoseLandmark.LEFT_HIP].y)
    if shoulder_hip_vertical_diff < 0.15:
        return "Sleeping"  # Return sleeping status if conditions are met

    # Check for jumping based on feet height
    feet_height = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y + 
                  landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y) / 2
    if feet_height < 0.8:
        return "Jumping"  # Return jumping status if conditions are met

    return "Unknown Pose"  # Return unknown pose if none of the conditions are met

def process_images(image_paths, pose_est=True):
    """Process multiple images and detect poses, arranging them in a grid layout."""
    try:
        num_images = len(image_paths)  # Get the number of images to process
        columns = 5  # Set a fixed number of columns for the grid layout
        rows = (num_images // columns) + (num_images % columns > 0)  # Calculate the number of rows needed

        fig, axes = plt.subplots(rows, columns, figsize=(15, 5 * rows))  # Create a grid of subplots
        
        # Flatten axes for easier indexing if multiple rows or columns
        if rows > 1 and columns > 1:
            axes = axes.flatten()  # Flatten the axes array for easier access
        elif rows == 1 or columns == 1:
            axes = np.array(axes).reshape(-1)  # Reshape if only one row or column

        fig.subplots_adjust(left=0.1, right=0.5, top=0.7, bottom=0.2, wspace=0.4, hspace=0.4)  # Adjust subplot layout

        for idx, image_path in enumerate(image_paths):
            # Read and validate image
            image = cv2.imread(image_path)  # Load the image from the specified path
            if image is None:
                raise Exception(f"Could not load image at {image_path}")  # Raise an error if the image cannot be loaded

            # Convert color and process the image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB format
            results = pose.process(image_rgb)  # Process the image to detect pose landmarks

            # Check if pose landmarks are detected
            if not results.pose_landmarks:
                pose_class = "No pose detected"  # Set pose class if no landmarks are found
            else:
                pose_class = classify_pose(results.pose_landmarks.landmark)  # Classify the pose based on landmarks
                annotated_image = image.copy()  # Create a copy of the original image for annotation
                mp_draw.draw_landmarks(
                    annotated_image, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,  # Draw connections between landmarks
                    mp_draw.DrawingSpec(color=(245,117,66), thickness=5, circle_radius=5),  # Style for landmarks
                    mp_draw.DrawingSpec(color=(245,66,230), thickness=5, circle_radius=5)  # Style for connections
                )
                image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)  # Convert annotated image to RGB

            # Display image in the appropriate subplot
            ax = axes[idx]  # Get the current axis for the subplot
            ax.imshow(image_rgb)  # Display the image
            ax.axis('off')  # Hide the axis
            if pose_est:
                ax.set_title(f'Pose: {pose_class}')  # Set the title to show detected pose

        # Hide any unused subplots
        for idx in range(num_images, rows * columns):
            fig.delaxes(axes[idx])  # Remove any unused axes from the figure

        plt.tight_layout(pad=2.0, w_pad=2.0, h_pad=8.0)  # Adjust layout for better spacing
        plt.savefig("output_img.png")  # Save the figure as an image
        plt.show()  # Display the figure

    except Exception as e:
        print(f"Error: {str(e)}")  # Print error message if an exception occurs