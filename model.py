#!/usr/bin/env python3
"""
AI Javelin Coach - Biomechanical Analysis Module
Command-line tool for analyzing javelin throw videos
"""

import sys
import json
import os
import cv2
import mediapipe as mp
import numpy as np
import math
from datetime import datetime

# Check if we have the minimum required arguments
if len(sys.argv) != 4:
    print("ERROR: Usage: python model.py <video_path> <height_m> <weight_kg>")
    sys.exit(1)

video_path = sys.argv[1]
try:
    athlete_height = float(sys.argv[2])
    athlete_weight = float(sys.argv[3])
except ValueError:
    print("ERROR: Height and weight must be numeric values")
    sys.exit(1)

print(f"Starting analysis for video: {os.path.basename(video_path)}")
print(f"Athlete parameters - Height: {athlete_height}m, Weight: {athlete_weight}kg")

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
    
    return angle

def extract_key_frame_angles(video_path):
    """Extract angles from key frames in the video"""
    print("Opening video file...")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("ERROR: Could not open video file")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has {total_frames} frames")
    
    # Analyze frames at different phases
    key_frames = {
        'approach': int(total_frames * 0.3),
        'crossover': int(total_frames * 0.6),
        'release': int(total_frames * 0.8)
    }
    
    results = {}
    
    for phase, frame_num in key_frames.items():
        print(f"Analyzing {phase} phase at frame {frame_num}")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            print(f"WARNING: Could not read frame {frame_num}")
            continue
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        pose_results = pose.process(rgb_frame)
        
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            
            # Extract key angles
            angles = {}
            
            try:
                # Right elbow angle
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
                
                angles['right_elbow'] = calculate_angle(right_shoulder, right_elbow, right_wrist)
                
                # Right shoulder angle
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
                angles['right_shoulder'] = calculate_angle(right_elbow, right_shoulder, right_hip)
                
                # Right knee angle
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
                angles['right_knee'] = calculate_angle(right_hip, right_knee, right_ankle)
                
                results[phase] = angles
                print(f"Extracted {len(angles)} angles for {phase} phase")
                
            except Exception as e:
                print(f"WARNING: Could not extract angles for {phase}: {str(e)}")
    
    cap.release()
    return results

def analyze_technique(angles_data):
    """Analyze technique and provide feedback"""
    print("Analyzing technique...")
    
    if not angles_data:
        return None
    
    # Simple analysis - find most deviant angles
    issues = []
    most_deviant_angle = None
    max_deviation = 0
    
    # Ideal angles (simplified)
    ideal_angles = {
        'approach': {'right_elbow': 160, 'right_shoulder': 45, 'right_knee': 150},
        'crossover': {'right_elbow': 165, 'right_shoulder': 50, 'right_knee': 140},
        'release': {'right_elbow': 170, 'right_shoulder': 160, 'right_knee': 160}
    }
    
    for phase, angles in angles_data.items():
        if phase in ideal_angles:
            for angle_name, measured_value in angles.items():
                if angle_name in ideal_angles[phase]:
                    ideal_value = ideal_angles[phase][angle_name]
                    deviation = abs(measured_value - ideal_value)
                    
                    if deviation > max_deviation:
                        max_deviation = deviation
                        most_deviant_angle = angle_name
                    
                    if deviation > 15:  # Significant deviation
                        direction = "too extended" if measured_value > ideal_value else "too flexed"
                        issues.append(f"{angle_name} in {phase} phase is {direction}")
    
    # Generate suggestions
    suggestions = []
    if most_deviant_angle:
        if 'elbow' in most_deviant_angle:
            suggestions.append("Focus on arm extension during the throw")
        elif 'shoulder' in most_deviant_angle:
            suggestions.append("Work on shoulder positioning and rotation")
        elif 'knee' in most_deviant_angle:
            suggestions.append("Improve leg drive and stability")
    
    if not suggestions:
        suggestions.append("Good technique! Continue practicing for consistency")
    
    return {
        'most_deviant_angle': most_deviant_angle or 'technique',
        'deviation': max_deviation,
        'issues': issues,
        'suggestions': suggestions
    }

def main():
    """Main analysis function"""
    try:
        # Check if video file exists
        if not os.path.exists(video_path):
            print("ERROR: Video file not found")
            sys.exit(1)
        
        print("Starting pose detection...")
        
        # Extract angles from video
        angles_data = extract_key_frame_angles(video_path)
        
        if not angles_data:
            print("ERROR: Could not extract pose data from video")
            sys.exit(1)
        
        print("Performing technique analysis...")
        
        # Analyze technique
        analysis = analyze_technique(angles_data)
        
        if not analysis:
            print("ERROR: Could not complete technique analysis")
            sys.exit(1)
        
        # Generate coaching feedback
        main_suggestion = analysis['suggestions'][0] if analysis['suggestions'] else "Keep practicing!"
        
        # Prepare final result
        result = {
            'status': 'success',
            'most_deviant_angle': analysis['most_deviant_angle'],
            'deviation': round(analysis['deviation'], 1),
            'llm_suggestion': main_suggestion,
            'detailed_analysis': {
                'issues': analysis['issues'],
                'all_suggestions': analysis['suggestions'],
                'angles_measured': angles_data
            }
        }
        
        print("Analysis completed successfully!")
        print("Generating coaching recommendations...")
        
        # Output final JSON result (this will be captured by Flask)
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            'status': 'error',
            'error': str(e),
            'most_deviant_angle': 'unknown',
            'deviation': 0,
            'llm_suggestion': 'Unable to analyze video. Please try again with a clearer video.'
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main()