# Flask Backend Server #HIT400
# Initialize heavy commenting!

import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import werkzeug.utils # For secure filenames
import traceback # For detailed error logging
import logging # For more structured logging

# Import necessary functions from your existing model.py
# Ensure model.py is in the same directory or adjust Python's import path
try:
    from model import analyze_throw, save_results_to_csv, processor as generate_llm_feedback
    # 'processor' from model.py is the function that calls FeedbackProcessor
except ImportError as e:
    print(f"Error importing from model.py: {e}")
    print("Make sure model.py is in the same directory and all its dependencies are installed.")
    exit()

try:
    import pandas as pd
except ImportError:
    print("Pandas is not installed. Please install it: pip install pandas")
    exit()

app = Flask(__name__)
CORS(app)

# --- Config --- #
UPLOAD_FOLDER = 'uploads' # Folder to temporarily store uploaded videos
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'} # Allowed video extensions
IDEAL_ANGLES_PATH = 'idealAngles.json' # Path to your ideal angles JSON
ANALYSIS_LOG_CSV = 'javelin_analysis_log.csv' # Path to the CSV log file

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Logging Setup ---
# More detailed logging. To save the logging file or not, that is the question...
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')


def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """Home route to check if the server is running."""
    return "Athlete Performance Tracker Backend is running!"

# --- Endpoint for Video Analysis ---
@app.route('/analyze', methods=['POST'])
def analyze_video_endpoint():
    """
    Receives a video file and athlete data, performs analysis,
    and returns the report and feedback.
    """
    video_path = None # Initialize to ensure it's defined for cleanup in case of error
    try:
        # --- 1. Check for video file ---
        if 'video' not in request.files:
            app.logger.error("No video file part in the request")
            return jsonify({"error": "No video file part in the request"}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            app.logger.error("No selected video file")
            return jsonify({"error": "No selected video file"}), 400

        # --- 2. Get athlete metadata from form data ---
        athlete_id = request.form.get('athlete_id')
        athlete_height_str = request.form.get('athlete_height')
        athlete_weight_str = request.form.get('athlete_weight')

        if not all([athlete_id, athlete_height_str, athlete_weight_str]):
            app.logger.error("Missing athlete_id, athlete_height, or athlete_weight in form data")
            return jsonify({"error": "Missing athlete_id, athlete_height, or athlete_weight in form data"}), 400

        try:
            athlete_height = float(athlete_height_str)
            athlete_weight = float(athlete_weight_str)
        except ValueError:
            app.logger.error(f"Invalid format for height ('{athlete_height_str}') or weight ('{athlete_weight_str}'). Must be numbers.")
            return jsonify({"error": "Invalid format for height or weight. Must be numbers."}), 400

        # --- 3. Securely save the uploaded video ---
        if video_file and allowed_file(video_file.filename):
            filename = werkzeug.utils.secure_filename(video_file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video_file.save(video_path)
            app.logger.info(f"Video saved to: {video_path}")
        else:
            app.logger.error(f"Invalid file type: {video_file.filename}. Allowed: {ALLOWED_EXTENSIONS}")
            return jsonify({"error": "Invalid file type. Allowed types are: " + ", ".join(ALLOWED_EXTENSIONS)}), 400

        # --- 4. Perform the analysis using your model.py functions ---
        app.logger.info(f"Analyzing video: {video_path} for athlete: {athlete_id}, H: {athlete_height}, W: {athlete_weight}")
        
        # Call analyze_throw from model.py
        # It expects: video_path, ideal_angles_path, athlete_height_m, athlete_weight_kg
        analysis_report_data = analyze_throw(video_path, IDEAL_ANGLES_PATH, athlete_height, athlete_weight)

        if not analysis_report_data:
            app.logger.error("Analysis failed or returned no data from analyze_throw.")
            # Clean up uploaded file if analysis fails
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
            return jsonify({"error": "Video analysis failed to produce results."}), 500

        # --- 5. Generate Gemini Feedback (using the processor function from model.py) ---
        # The processor function in model.py uses FeedbackProcessor, which expects
        # the LM Studio API to be running at http://localhost:1234/v1/chat/completions The LM studio thing did not work, so we go for Gemini
        app.logger.info("Generating LLM feedback...")
        llm_feedback = generate_llm_feedback(analysis_report_data) # This is the 'processor' function
        if llm_feedback is None:
            app.logger.warning("LLM feedback generation failed or returned None. Proceeding without it.")
            llm_feedback = "Automated feedback could not be generated at this time."


        # --- 6. Save results to CSV (using function from model.py) ---
        # The save_results_to_csv function expects:
        # analysis_report, athlete_id, video_filename, csv_filepath
        save_results_to_csv(analysis_report_data, athlete_id, filename, ANALYSIS_LOG_CSV)
        app.logger.info(f"Analysis results saved to CSV: {ANALYSIS_LOG_CSV}")

        # --- 7. Clean up the uploaded video file ---
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
            app.logger.info(f"Cleaned up uploaded file: {video_path}")
            video_path = None # Reset path after deletion

        # --- 8. Return the results ---
        app.logger.info("Analysis successful. Returning results to client.")
        return jsonify({
            "message": "Analysis successful",
            "athlete_id": athlete_id,
            "report": analysis_report_data, # This is the dict from analyze_throw
            "feedback": llm_feedback
        }), 200

    except Exception as e:
        app.logger.error(f"An error occurred during /analyze: {traceback.format_exc()}")
        # Clean up if video_path was defined and file exists
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
            app.logger.info(f"Cleaned up uploaded file due to error: {video_path}")
        return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500


@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Process the uploaded file here
    return jsonify({"message": "File uploaded successfully"}), 200


# --- Endpoint for Fetching Athlete's Analysis History ---
@app.route('/history/<string:athlete_id>', methods=['GET'])
def get_athlete_history(athlete_id):
    """
    Retrieves the analysis history for a given athlete_id from the CSV log.
    """
    try:
        if not os.path.exists(ANALYSIS_LOG_CSV):
            app.logger.info(f"Analysis log CSV not found ({ANALYSIS_LOG_CSV}) for history request: athlete {athlete_id}")
            return jsonify({"message": "No analysis history found for any athlete yet."}), 404

        df = pd.read_csv(ANALYSIS_LOG_CSV)
        
        # Convert timestamp to a consistent string format for JSON and sorting
        # Also create a datetime object for proper sorting if not already
        if 'timestamp' in df.columns:
            try:
                df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
                df['timestamp_str'] = df['timestamp_dt'].dt.strftime('%Y-%m-%dT%H:%M:%S') # ISO 8601 like format
            except Exception as e:
                app.logger.warning(f"Could not parse timestamps for history: {e}. Using raw timestamps for sorting and output.")
                # If parsing fails, assume timestamp is already a sortable string or handle as best as possible
                df['timestamp_dt'] = df['timestamp'] # May not sort correctly if mixed formats
                df['timestamp_str'] = df['timestamp'].astype(str)
        else:
            app.logger.warning("Timestamp column missing in CSV. History might be incomplete or unsorted.")
            return jsonify({"error": "Timestamp data missing in history log."}), 500


        athlete_history_df = df[df['athlete_id'] == athlete_id].copy() # Use .copy() to avoid SettingWithCopyWarning

        if athlete_history_df.empty:
            app.logger.info(f"No history found for athlete_id: {athlete_id} in {ANALYSIS_LOG_CSV}")
            return jsonify({"message": f"No analysis history found for athlete_id: {athlete_id}"}), 404
        
        # Ensure 'timestamp_dt' exists for sorting, even if it's a copy of 'timestamp'
        if 'timestamp_dt' not in athlete_history_df.columns and 'timestamp' in athlete_history_df.columns:
             athlete_history_df['timestamp_dt'] = pd.to_datetime(athlete_history_df['timestamp'], errors='coerce')


        history_list = []
        # Group by the original analysis session (timestamp from the log) and video file
        # Use the parsed datetime for grouping if available, otherwise string
        # Sort before grouping to ensure consistent session identification if timestamps are very close
        if 'timestamp_dt' in athlete_history_df.columns:
            athlete_history_df.sort_values(by='timestamp_dt', inplace=True)
            grouping_cols = ['timestamp_dt', 'video_file']
        else: # Fallback if timestamp_dt couldn't be created
            athlete_history_df.sort_values(by='timestamp', inplace=True)
            grouping_cols = ['timestamp', 'video_file']


        for group_keys, group_df in athlete_history_df.groupby(grouping_cols):
            # Get the string version of the timestamp for output
            session_timestamp_str = group_df['timestamp_str'].iloc[0] if 'timestamp_str' in group_df.columns else str(group_keys[0])
            video_name = group_keys[1]
            
            report_data_for_session = {}
            # Reconstruct the 'report' structure for this session
            for _, row in group_df.iterrows():
                phase = row['phase']
                angle_name = row['angle_name']
                if phase not in report_data_for_session:
                    report_data_for_session[phase] = {}
                report_data_for_session[phase][angle_name] = {
                    'calculated': row['calculated'] if pd.notna(row['calculated']) else None,
                    'ideal': row['ideal'] if pd.notna(row['ideal']) else None,
                    'deviation': row['deviation'] if pd.notna(row['deviation']) else None
                }
            
            # For this simple version, we don't have the original LLM feedback stored per historical entry.
            # The CSV log only stores angle data. If feedback needs to be retrieved historically,
            # it would need to be saved alongside the angle data in the CSV or another store.
            # For now, feedback for historical entries will be missing.
            history_list.append({
                "timestamp": session_timestamp_str, # Use the consistent string version
                "videoName": video_name,
                "reportData": report_data_for_session,
                "feedback": "Feedback for historical entries is not stored in this version." 
            })
            
        # Sort by timestamp descending (most recent first) using the string timestamp
        history_list.sort(key=lambda x: x['timestamp'], reverse=True)
        app.logger.info(f"Retrieved {len(history_list)} history entries for athlete {athlete_id}")
        return jsonify(history_list), 200

    except pd.errors.EmptyDataError:
         app.logger.info(f"Analysis log CSV ({ANALYSIS_LOG_CSV}) is empty for history request: athlete {athlete_id}")
         return jsonify({"message": "Analysis log is empty or not correctly formatted."}), 404
    except Exception as e:
        app.logger.error(f"An error occurred during /history for athlete {athlete_id}: {traceback.format_exc()}")
        return jsonify({"error": "An internal server error occurred while fetching history.", "details": str(e)}), 500


if __name__ == '__main__':
    # Enable Flask's built-in debugger and reloader for development
    # Important: DO NOT run with debug=True in a production environment!
    # Use host='0.0.0.0' to make the server accessible from other devices on the same network
    app.run(host='0.0.0.0', port=5000, debug=True)