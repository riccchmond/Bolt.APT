# Import necessary libraries
import cv2
import mediapipe as mp
import numpy as np
import json
import math
import os
import feedback
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision.io import read_video
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from datetime import datetime
from feedback1 import FeedbackProcessor

# --- Mediapipe Setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Angle Calculation Function ---
def calculate_angle(a, b, c):
    """Calculates the angle between three points (e.g., elbow angle from shoulder, elbow, wrist)."""
    a = np.array(a) # First point
    b = np.array(b) # Mid point (vertex)
    c = np.array(c) # End point

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle

# --- Landmark Extraction Function ---
# (No changes needed in this function)
def extract_landmarks(video_path):
    """Processes video, extracts landmarks and calculates angles frame by frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    frame_data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # End of video

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Convert back to BGR for drawing
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates - Ensure landmarks are visible
            coords = {lm_name: [landmarks[mp_pose.PoseLandmark[lm_name].value].x,
                                landmarks[mp_pose.PoseLandmark[lm_name].value].y,
                                landmarks[mp_pose.PoseLandmark[lm_name].value].z, # Might need to use this later
                                landmarks[mp_pose.PoseLandmark[lm_name].value].visibility]
                      for lm_name in mp_pose.PoseLandmark.__members__ if landmarks[mp_pose.PoseLandmark[lm_name].value].visibility > 0.3} # Filter by visibility
            #frame_data.append(coords)

            # Calculate angles if landmarks are available
            angles = {}
            # Define landmarks needed for each angle
            angle_definitions = {
                'left_elbow': ('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'),
                'right_elbow': ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST'),
                'left_shoulder': ('LEFT_ELBOW', 'LEFT_SHOULDER', 'LEFT_HIP'),
                'right_shoulder': ('RIGHT_ELBOW', 'RIGHT_SHOULDER', 'RIGHT_HIP'),
                'left_hip': ('LEFT_KNEE', 'LEFT_HIP', 'LEFT_SHOULDER'),
                'right_hip': ('RIGHT_KNEE', 'RIGHT_HIP', 'RIGHT_SHOULDER'),
                'left_knee': ('LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'),
                'right_knee': ('RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'),
                'left_trunk': ('LEFT_HIP', 'LEFT_SHOULDER', 'RIGHT_SHOULDER'),  # Example proxy
                'right_trunk': ('RIGHT_HIP', 'RIGHT_SHOULDER', 'LEFT_SHOULDER')  # Example proxy
            }

            for angle_name, (p1_name, p2_name, p3_name) in angle_definitions.items():
                # Check if all required landmarks are visible before calculation
                if all(lm in coords for lm in [p1_name, p2_name, p3_name]):
                    p1 = coords[p1_name][:2]  # Use only x, y for 2D angle
                    p2 = coords[p2_name][:2]
                    p3 = coords[p3_name][:2]
                    angles[angle_name] = calculate_angle(p1, p2, p3)
                else:
                    angles[angle_name] = None  # Indicate angle couldn't be calculated

            # Store data for this frame
            frame_data.append({'frame': frame_count, 'landmarks': coords, 'angles': angles})

        except AttributeError:
            # No landmarks detected in this frame
            frame_data.append({'frame': frame_count, 'landmarks': None, 'angles': None})

            # --- Optional: Draw landmarks for visualization ---
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow('Mediapipe Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            # --- End Optional Visualization ---

        frame_count += 1

    cap.release()
        # cv2.destroyAllWindows() # Close window if imshow was used
    print(f"Processed {frame_count} frames.")
    return frame_data

    # --- Ideal Angle Loading and Interpolation ---

    # *** CHANGED FUNCTION ***
def load_ideal_angles(json_path):
    """Loads the ideal angle data from the *structured* JSON file."""
    try:
        with open(json_path, 'r') as f:
            # Load the already structured JSON data directly
            data = json.load(f)
            print(f"Loaded structured ideal angles data from {json_path}.")
            return data
    except FileNotFoundError:
        print(f"Error: Ideal angles JSON file not found at {json_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return None
    except Exception as e:
        print(f"An error occurred loading ideal angles: {e}")
        return None


# *** CHANGED FUNCTION ***
def get_ideal_angle(ideal_angles_data, phase, athlete_height, athlete_weight):
    """
    Gets the ideal angles for a specific phase, height, and weight
    using the structured JSON data and bilinear interpolation (height first, then weight).
    """
    if ideal_angles_data is None or phase not in ideal_angles_data or 'data_points' not in ideal_angles_data[phase]:
        print(f"Warning: Phase '{phase}' or its data points not found in ideal angles data.")
        return None

    data_points = ideal_angles_data[phase]['data_points']
    if not data_points:
        print(f"Warning: No data points found for phase '{phase}'.")
        return None

    # --- Find bracketing weights ---
    unique_weights = sorted(list(set(dp['weight'] for dp in data_points)))
    if not unique_weights:
        print(f"Warning: No valid weight data found in phase '{phase}'.")
        return None

    w1, w2 = None, None
    if athlete_weight <= unique_weights[0]:
        w1 = w2 = unique_weights[0]  # Below or at minimum weight
        print(f"Athlete weight {athlete_weight}kg <= minimum ({w1}kg). Using minimum weight data.")
    elif athlete_weight >= unique_weights[-1]:
        w1 = w2 = unique_weights[-1]  # Above or at maximum weight
        print(f"Athlete weight {athlete_weight}kg >= maximum ({w1}kg). Using maximum weight data.")
    else:
        # Find weights that bracket the athlete's weight
        for i in range(len(unique_weights) - 1):
            if unique_weights[i] <= athlete_weight < unique_weights[i + 1]:
                w1 = unique_weights[i]
                w2 = unique_weights[i + 1]
                break
        if w1 is None:  # Should not happen if logic above is correct, but as failsafe
            w1 = w2 = unique_weights[-1]
            print(f"Warning: Could not bracket weight {athlete_weight}kg. Using max weight {w1}kg.")

    print(f"Using weights {w1}kg and {w2}kg to interpolate/extrapolate for athlete weight {athlete_weight}kg.")

    # --- Helper function for height interpolation at a specific weight ---
    def interpolate_height(target_height, weight_value):
        # Filter data points for the specific weight
        points_at_weight = [dp for dp in data_points if dp['weight'] == weight_value]
        if not points_at_weight:
            return None  # No data for this specific weight

        # Sort points by height
        points_at_weight.sort(key=lambda x: x['height'])
        heights = np.array([p['height'] for p in points_at_weight])

        # Get names of angles available in the first point (assume consistent)
        angle_names = list(points_at_weight[0]['angles'].keys())
        interpolated_angles = {}

        min_h, max_h = heights.min(), heights.max()

        for angle_name in angle_names:
            # Ensure angle exists in all points for this weight (can happen with sparse data)
            if not all(angle_name in p['angles'] for p in points_at_weight):
                print(f"Warning: Angle '{angle_name}' missing in some data points for weight {weight_value}. Skipping.")
                interpolated_angles[angle_name] = None
                continue

            values = np.array([p['angles'][angle_name] for p in points_at_weight])

            # Perform interpolation/extrapolation based on height
            if target_height < min_h:
                interp_val = np.interp(target_height, heights[:2], values[:2]) if len(heights) >= 2 else values[0]
            elif target_height > max_h:
                interp_val = np.interp(target_height, heights[-2:], values[-2:]) if len(heights) >= 2 else values[-1]
            else:
                interp_val = np.interp(target_height, heights, values)

            interpolated_angles[angle_name] = interp_val

        return interpolated_angles

    # --- Perform height interpolation for the bracketing weights ---
    angles_w1 = interpolate_height(athlete_height, w1)
    angles_w2 = None
    if w1 != w2:  # Only need second interpolation if weights are different
        angles_w2 = interpolate_height(athlete_height, w2)

    if angles_w1 is None:
        print(f"Error: Could not get interpolated angles for weight {w1}.")
        return None
    if w1 != w2 and angles_w2 is None:
        print(f"Warning: Could not get interpolated angles for weight {w2}. Using results from weight {w1}.")
        return angles_w1  # Fallback to angles from the lower weight

    # --- Interpolate between the two weight results ---
    final_angles = {}
    angle_names = list(angles_w1.keys())  # Assume keys are consistent

    for angle_name in angle_names:
        val1 = angles_w1.get(angle_name)

        if w1 == w2 or angles_w2 is None:  # Use w1 results if extrapolating or w2 failed
            final_angles[angle_name] = val1
        else:
            val2 = angles_w2.get(angle_name)
            # Check if both values are valid for interpolation
            if val1 is not None and val2 is not None:
                # Linear interpolation between the two weight results
                # interp_factor = (athlete_weight - w1) / (w2 - w1)
                # final_angles[angle_name] = val1 + interp_factor * (val2 - val1)
                # Using numpy's interp for consistency
                final_angles[angle_name] = np.interp(athlete_weight, [w1, w2], [val1, val2])

            elif val1 is not None:  # If only val1 is valid, use it
                final_angles[angle_name] = val1
                print(f"Warning: Missing value for '{angle_name}' at weight {w2}. Using value from weight {w1}.")
            else:  # If neither (or only val2) is valid
                final_angles[angle_name] = val2  # Use val2 if available, otherwise None
                if val2 is None:
                    print(f"Warning: Missing values for '{angle_name}' at both weights {w1} and {w2}.")

    return final_angles


# --- Heuristic Phase Identification ---
# (No changes needed in this function)
def identify_key_phases(frame_data):
    """Identifies key frames for 'crossover' and 'release' using heuristics."""
    key_frames = {'crossover': None, 'release': None}
    if not frame_data or len(frame_data) < 10:  # Need sufficient frames
        print("Warning: Insufficient frame data for phase identification.")
        return key_frames

    max_velocity = -1
    release_frame_index = -1
    target_landmark = 'RIGHT_ELBOW'  # Placeholder - needs dominant arm logic

    for i in range(1, len(frame_data)):
        prev_frame = frame_data[i - 1]
        curr_frame = frame_data[i]

        if curr_frame['landmarks'] and prev_frame['landmarks'] and \
                target_landmark in curr_frame['landmarks'] and \
                target_landmark in prev_frame['landmarks']:

            prev_pos = np.array(prev_frame['landmarks'][target_landmark][:2])
            curr_pos = np.array(curr_frame['landmarks'][target_landmark][:2])
            velocity = np.linalg.norm(curr_pos - prev_pos)

            if velocity > max_velocity and i > len(frame_data) // 2:
                max_velocity = velocity
                release_frame_index = i

    if release_frame_index != -1:
        key_frames['release'] = release_frame_index
        print(f"Identified potential release frame: {release_frame_index}")

    crossover_frame_index = -1
    if release_frame_index != -1:
        potential_crossover_index = int(release_frame_index * 0.7)  # Rough estimate
        if 0 <= potential_crossover_index < len(frame_data) and frame_data[potential_crossover_index]['angles']:
            crossover_frame_index = potential_crossover_index
            key_frames['crossover'] = crossover_frame_index  # Map to 'step_1' or 'step_2' based on JSON
            print(f"Identified potential crossover frame: {crossover_frame_index} (mapped to 'step_1')")
        else:
            mid_point = len(frame_data) // 2
            if frame_data[mid_point] and frame_data[mid_point]['angles']:
                crossover_frame_index = mid_point
                key_frames['crossover'] = crossover_frame_index
                print(f"Identified potential crossover frame (fallback): {crossover_frame_index} (mapped to 'step_1')")

    if key_frames['crossover'] is not None:
        # *** Map crossover heuristic frame to 'step_1' for JSON lookup ***
        key_frames['step_1'] = key_frames.pop('crossover')

    if key_frames.get('release') is None: print("Warning: Could not identify release frame.")
    if key_frames.get('step_1') is None: print("Warning: Could not identify crossover ('step_1') frame.")

    return key_frames


# --- Dominant Arm Detection ---
# (No changes needed in this function)
# --- Dominant Arm Detection (Revised) ---
def detect_dominant_arm(frame_data, key_frames):
    """
    Heuristic to detect dominant (throwing) arm based on peak wrist velocity
    and forward leg position at release.
    """
    right_score = 0
    left_score = 0

    if not frame_data or len(frame_data) < 5:
        print("Warning: Insufficient data for dominant arm detection. Defaulting to right.")
        return 'right'

    release_frame_index = key_frames.get('release')

    # --- Heuristic 1: Peak Wrist Velocity (Leading up to Release) ---
    max_right_wrist_vel = 0
    max_left_wrist_vel = 0
    # Define window: e.g., from 50% of way to release up to release frame + small buffer
    start_vel_check = int(len(frame_data) * 0.5)
    end_vel_check = len(frame_data)
    if release_frame_index is not None:
        start_vel_check = max(0, int(release_frame_index * 0.6)) # Start earlier before release
        end_vel_check = min(len(frame_data) -1, release_frame_index + 3) # Check slightly past release too

    for i in range(start_vel_check + 1, end_vel_check + 1) :
        if i >= len(frame_data): break # Ensure index is within bounds

        prev_frame = frame_data[i-1]
        curr_frame = frame_data[i]

        # Check right wrist velocity
        if curr_frame.get('landmarks') and prev_frame.get('landmarks') and \
           'RIGHT_WRIST' in curr_frame['landmarks'] and 'RIGHT_WRIST' in prev_frame['landmarks']:
           prev_pos = np.array(prev_frame['landmarks']['RIGHT_WRIST'][:2])
           curr_pos = np.array(curr_frame['landmarks']['RIGHT_WRIST'][:2])
           velocity = np.linalg.norm(curr_pos - prev_pos)
           if velocity > max_right_wrist_vel:
               max_right_wrist_vel = velocity

        # Check left wrist velocity
        if curr_frame.get('landmarks') and prev_frame.get('landmarks') and \
           'LEFT_WRIST' in curr_frame['landmarks'] and 'LEFT_WRIST' in prev_frame['landmarks']:
           prev_pos = np.array(prev_frame['landmarks']['LEFT_WRIST'][:2])
           curr_pos = np.array(curr_frame['landmarks']['LEFT_WRIST'][:2])
           velocity = np.linalg.norm(curr_pos - prev_pos)
           if velocity > max_left_wrist_vel:
               max_left_wrist_vel = velocity

    # Award point based on velocity (require a significant difference to avoid noise)
    if max_right_wrist_vel > max_left_wrist_vel * 1.3: # Right significantly faster
        right_score += 1
        print("Dominant Arm Heuristic: Right wrist velocity higher.")
    elif max_left_wrist_vel > max_right_wrist_vel * 1.3: # Left significantly faster
        left_score += 1
        print("Dominant Arm Heuristic: Left wrist velocity higher.")
    else:
        print("Dominant Arm Heuristic: Wrist velocities inconclusive.")


    # --- Heuristic 2: Forward Leg at Release ---
    if release_frame_index is not None and \
       frame_data[release_frame_index].get('landmarks'):
        landmarks_at_release = frame_data[release_frame_index]['landmarks']
        if 'LEFT_ANKLE' in landmarks_at_release and 'RIGHT_ANKLE' in landmarks_at_release:
            left_ankle_x = landmarks_at_release['LEFT_ANKLE'][0]
            right_ankle_x = landmarks_at_release['RIGHT_ANKLE'][0]

            # Assuming side view where larger X is "forward" relative to camera direction
            # This might need adjustment if camera faces the other way or view is not side-on
            # A more robust check might involve position relative to the hip center.
            if left_ankle_x > right_ankle_x: # Left leg is forward -> Right arm dominant
                right_score += 1
                print("Dominant Arm Heuristic: Left leg forward at release.")
            elif right_ankle_x > left_ankle_x: # Right leg is forward -> Left arm dominant
                left_score += 1
                print("Dominant Arm Heuristic: Right leg forward at release.")
            else:
                print("Dominant Arm Heuristic: Leg positions inconclusive at release.")
        else:
            print("Dominant Arm Heuristic: Ankle landmarks missing at release frame.")
    else:
        print("Dominant Arm Heuristic: Release frame not identified or landmarks missing, cannot use leg position.")


    # --- Determine Dominant Arm ---
    print(f"Dominant Arm Scores - Left: {left_score}, Right: {right_score}")
    if left_score > right_score:
        return 'left'
    elif right_score > left_score:
        return 'right'
    else:
        # If scores are tied (or both 0), default to right (or could use velocity as tie-breaker if needed)
        print("Dominant Arm Heuristic: Scores tied or inconclusive. Defaulting to right.")
        return 'right'

# You would call this function within `analyze_throw` *after* identifying key frames:
# dominant_arm_side = detect_dominant_arm(frame_data, key_frames)


# --- Analysis Function ---
# (Minor change: Pass correct ideal_angles_data structure)
def analyze_throw(video_path, ideal_angles_path, athlete_height_m, athlete_weight_kg):
    """Main function to analyze the javelin throw video."""

    print(f"Analyzing video: {video_path}")
    print(f"Athlete - Height: {athlete_height_m}m, Weight: {athlete_weight_kg}kg")

    # 1. Extract Landmarks and Angles
    frame_data = extract_landmarks(video_path)
    if not frame_data:
        print("Analysis failed: Could not process video.")
        return None

    # 2. Identify Key Phases (Heuristic)
    key_frames = identify_key_phases(frame_data)  # Returns dict like {'step_1': frame_idx, 'release': frame_idx}

    # 3. Detect Dominant Arm (Placeholder)
    dominant_arm_side = detect_dominant_arm(frame_data, key_frames)  # Returns 'left' or 'right'


    # 4. Load Ideal Angles (Using the modified loader)
    ideal_angles_data = load_ideal_angles(ideal_angles_path)
    if ideal_angles_data is None:
        print("Analysis failed: Could not load ideal angles.")
        return None

    # 5. Compare Angles and Generate Report
    analysis_report = {}

    for phase_name, frame_index in key_frames.items():
        if frame_index is None:
            print(f"Skipping comparison for phase '{phase_name}' (frame not identified).")
            continue
        if frame_index >= len(frame_data) or frame_data[frame_index]['angles'] is None:
            print(f"Skipping comparison for phase '{phase_name}' (no angle data at frame {frame_index}).")
            continue

        print(f"\n--- Analyzing Phase: {phase_name} (Frame: {frame_index}) ---")

        calculated_angles = frame_data[frame_index]['angles']
        dominant_angles = {}
        for angle_name, value in calculated_angles.items():
            if angle_name == f"{dominant_arm_side}_elbow":
                dominant_angles['dominant_elbow'] = value
            elif angle_name == f"{dominant_arm_side}_shoulder":
                dominant_angles['dominant_shoulder'] = value
            elif angle_name.startswith(dominant_arm_side):
                dominant_angles[angle_name] = value
            elif angle_name.startswith('left') and dominant_arm_side == 'right':
                dominant_angles[angle_name] = value
            elif angle_name.startswith('right') and dominant_arm_side == 'left':
                dominant_angles[angle_name] = value
            elif 'trunk' in angle_name:
                dominant_angles[angle_name] = value

        if not dominant_angles:
            print(f"Could not extract dominant angles for phase '{phase_name}'.")
            continue

        # Get ideal angles using interpolation (Using the modified function)
        ideal_phase_angles = get_ideal_angle(ideal_angles_data, phase_name, athlete_height_m, athlete_weight_kg)

        if ideal_phase_angles is None:
            print(f"Could not determine ideal angles for phase '{phase_name}'.")
            continue

        phase_report = {}
        print("Comparing Calculated vs Ideal Angles:")
        # Use angle names from the ideal data as the reference
        for angle_name, ideal_value in ideal_phase_angles.items():
            calculated_value = dominant_angles.get(angle_name)  # Get corresponding calculated angle

            if calculated_value is not None and ideal_value is not None:
                deviation = calculated_value - ideal_value
                print(
                    f"  {angle_name}: Calculated={calculated_value:.1f}°, Ideal={ideal_value:.1f}°, Deviation={deviation:.1f}°")
                phase_report[angle_name] = {
                    'calculated': round(calculated_value, 1),
                    'ideal': round(ideal_value, 1),
                    'deviation': round(deviation, 1)
                }
            elif ideal_value is not None:  # Only ideal is available
                print(f"  {angle_name}: Calculated=N/A, Ideal={ideal_value:.1f}° (Cannot compare)")
                phase_report[angle_name] = {
                    'calculated': None,
                    'ideal': round(ideal_value, 1),
                    'deviation': None
                }
            # If ideal_value is None, skip (already handled by skipping angle in loop)

        analysis_report[phase_name] = phase_report

    return analysis_report

# --- Helper Function to Save Results ---
def save_results_to_csv(analysis_report, athlete_id, video_filename, csv_filepath='javelin_analysis_log.csv'):
    """Appends the analysis results to a CSV file."""
    if not analysis_report:
        print("No analysis report to save.")
        return

    rows = []
    analysis_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Record when analysis was run

    for phase, phase_data in analysis_report.items():
        for angle_name, angle_values in phase_data.items():
            rows.append({
                'timestamp': analysis_timestamp,
                'athlete_id': athlete_id,
                'video_file': os.path.basename(video_filename), # Store only filename
                'phase': phase,
                'angle_name': angle_name,
                'calculated': angle_values.get('calculated'),
                'ideal': angle_values.get('ideal'),
                'deviation': angle_values.get('deviation')
            })

    # Create DataFrame
    df_new = pd.DataFrame(rows)

    # Append to CSV file, creating it if it doesn't exist
    if os.path.exists(csv_filepath):
        df_new.to_csv(csv_filepath, mode='a', header=False, index=False)
        print(f"Appended results to {csv_filepath}")
    else:
        df_new.to_csv(csv_filepath, mode='w', header=True, index=False)
        print(f"Created results file {csv_filepath}")

# --- LLM Integration ---
def processor(analysis_report):
    feedback_processor = FeedbackProcessor(api_key="AIzaSyD8No-aiDi_cbGRhGId-3liLNBL282pqjA")
    llmfeedback = feedback_processor.generate_coaching_feedback(analysis_report)
    if llmfeedback:
        print("\n--- Generated Feedback ---")
        print(llmfeedback)

    return llmfeedback
    
# --- Video Classification Function (NEW) ---

def classify_video_action(video_path, expected_action='javelin throw'):
    """
    Classifies the main action in a video using a pre-trained model.

    Args:
        video_path (str): The path to the video file.
        expected_action (str): The name of the action to check for.

    Returns:
        bool: True if the video is classified as the expected action, False otherwise.
    """
    print("\n--- Starting Video Classification ---")
    try:
        # 1. Load the pre-trained model and weights
        weights = R2Plus1D_18_Weights.KINETICS400_V1
        model = r2plus1d_18(weights=weights)
        model.eval() # Set model to evaluation mode

        # 2. Get the model's transformation pipeline and class names
        preprocess = weights.transforms()
        class_names = weights.meta["categories"]

        # 3. Read the video file
        # read_video reads the video and returns frames, audio frames, and metadata
        # We need to specify a start and end time in pts (presentation timestamp)
        # For simplicity, we'll read the first 150 frames.
        frames, _, _ = read_video(video_path, end_pts=150, pts_unit='frame')
        if frames.shape[0] == 0:
            print("Warning: Could not read frames from video for classification.")
            return False # Treat as not a javelin throw if video is unreadable

        # 4. Preprocess the video frames
        # The model expects a tensor of shape (B, C, T, H, W)
        # B=batch, C=channels, T=time/frames, H=height, W=width
        preprocessed_frames = preprocess(frames.permute(3, 0, 1, 2)).unsqueeze(0)

        # 5. Make a prediction
        with torch.no_grad():
            prediction = model(preprocessed_frames)

        # 6. Interpret the prediction
        # The output is a tensor of scores for each class.
        # We apply softmax to get probabilities and find the class with the highest score.
        probabilities = torch.nn.functional.softmax(prediction[0], dim=0)
        top_prob, top_cat_id = torch.topk(probabilities, 1)
        
        predicted_action = class_names[top_cat_id[0]]
        confidence = top_prob[0].item()

        print(f"Predicted action: '{predicted_action}' with {confidence:.2%} confidence.")

        # 7. Check if the prediction matches the expected action
        if predicted_action.lower() == expected_action.lower() and confidence > 0.5: # Confidence threshold
            print("Classification successful: Video is a javelin throw.")
            return True
        else:
            print(f"Warning: Video is NOT classified as a javelin throw. Found '{predicted_action}' instead.")
            return False

    except Exception as e:
        print(f"An error occurred during video classification: {e}")
        print("Skipping classification and proceeding with caution.")
        return True # Fail open: if classification fails, assume it's correct to avoid blocking analysis

# def processor(analysis_report):
#     feedback_processor = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=AIzaSyD8No-aiDi_cbGRhGId-3liLNBL282pqjA"
#     #feedback_processor = FeedbackProcessor(api_url="http://localhost:1234/v1/chat/completions", timeout=120)
#     llmfeedback = feedback_processor.generate_coaching_feedback(analysis_report)
#     if feedback:
#         print("\n--- Generated Feedback ---")
#         print(feedback)
#
#     return llmfeedback


# def generate_human_readable_feedback(analysis_report):
#     """Takes the analysis report and formats it for an LLM."""
#     if not analysis_report:
#         return "No analysis data to generate feedback."
#
#     prompt_lines = [
#         "Analyze the following javelin throw technical flaws and provide simple, actionable feedback for each phase:",
#         ""]
#
#     for phase, report in analysis_report.items():
#         prompt_lines.append(f"Phase: {phase}")
#         significant_deviations = False
#         for angle, data in report.items():
#             # Check if deviation exists and is significant
#             if data.get('deviation') is not None and abs(data['deviation']) > 5:  # Example threshold: 5 degrees
#                 significant_deviations = True
#                 deviation_desc = "lower" if data['deviation'] < 0 else "higher"
#                 # Ensure calculated and ideal values exist for the report string
#                 calc_str = f"{data['calculated']:.1f}°" if data['calculated'] is not None else "N/A"
#                 ideal_str = f"{data['ideal']:.1f}°" if data['ideal'] is not None else "N/A"
#                 prompt_lines.append(
#                     f"- {angle}: Is {abs(data['deviation']):.1f}° {deviation_desc} than ideal ({calc_str} vs {ideal_str}).")
#
#         if not significant_deviations:
#             prompt_lines.append("- No significant angle deviations detected.")
#         prompt_lines.append("")  # Add spacing
#
#     prompt = "\n".join(prompt_lines)
#
#     print("\n--- Generated Prompt for LLM ---")
#     print(prompt)
#     print("--- End LLM Prompt ---")
#
#     # Placeholder for LLM call
#     return "LLM integration placeholder. The prompt generated above should be sent to the LLM."


# --- Main Execution ---
if __name__ == "__main__":

    # --- Config ---
    video_file = r'D:\APT_DataSets\Javelin\good\Final\jav_64.mp4'  # Video file path
    ideal_angles_file = 'idealAngles.json'  # JSON file with ideal angles data
    athlete_height = 1.85  # meters
    athlete_weight = 88  # kilograms
    current_athlete_id = 'athlete_001'

    # Check if files exist
    if not os.path.exists(video_file):
        print(f"Error: Video file not found at {video_file}")
        exit()
    if not os.path.exists(ideal_angles_file):
        print(f"Error: Ideal angles JSON file not found at {ideal_angles_file}")
        exit()

    # --- NEW: Perform Video Classification First ---
    is_javelin_video = classify_video_action(video_file, expected_action='javelin throw')

    if not is_javelin_video:
        print("\n--- Analysis Halted ---")
        print("The provided video was not identified as a javelin throw. Exiting.")
        exit()
    # -----------------------------------------------

    print("\n--- Classification Confirmed: Starting Biomechanical Analysis ---")
    
    # --- Run Analysis (Original code proceeds from here) ---
    report = analyze_throw(video_file, ideal_angles_file, athlete_height, athlete_weight)

    # --- Generate Feedback ---
    if report:
        llm_feedback = processor(report)
        print("\n--- LLM Feedback ---")
        print(llm_feedback)

        save_results_to_csv(report, current_athlete_id, video_file, csv_filepath='javelin_analysis_log.csv')
    else:
        print("\nAnalysis could not be completed.")
