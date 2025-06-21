import os
import sys
import json
import uuid
import subprocess
import threading
import queue
from datetime import datetime
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from werkzeug.utils import secure_filename
import requests

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Configuration
UPLOAD_FOLDER = 'temp_videos'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}
TAVUS_API_URL = "https://api.tavus.io/v2/videos"
TAVUS_API_KEY = os.getenv('TAVUS_API_KEY', 'your_tavus_api_key_here')
REPLICA_ID = os.getenv('REPLICA_ID', 'your_replica_id_here')

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def stream_subprocess_output(process, output_queue):
    """Stream subprocess output to queue"""
    try:
        for line in iter(process.stdout.readline, b''):
            if line:
                output_queue.put(('stdout', line.decode('utf-8').strip()))
        
        for line in iter(process.stderr.readline, b''):
            if line:
                output_queue.put(('stderr', line.decode('utf-8').strip()))
                
        process.wait()
        output_queue.put(('done', process.returncode))
    except Exception as e:
        output_queue.put(('error', str(e)))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    def generate():
        video_path = None
        try:
            # Validate request
            if 'video' not in request.files:
                yield f"data: [ERROR] No video file provided\n\n"
                return
            
            video_file = request.files['video']
            height = request.form.get('height')
            weight = request.form.get('weight')
            
            if not all([video_file, height, weight]):
                yield f"data: [ERROR] Missing required fields\n\n"
                return
            
            if video_file.filename == '':
                yield f"data: [ERROR] No file selected\n\n"
                return
            
            if not allowed_file(video_file.filename):
                yield f"data: [ERROR] Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}\n\n"
                return
            
            # Save uploaded file
            filename = secure_filename(video_file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            video_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            video_file.save(video_path)
            
            yield f"data: [UPLOAD] Video saved successfully: {filename}\n\n"
            yield f"data: [INFO] Starting analysis for athlete (Height: {height}m, Weight: {weight}kg)\n\n"
            
            # Start analysis subprocess
            cmd = [sys.executable, 'model.py', video_path, height, weight]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=False,
                bufsize=1
            )
            
            # Stream output
            output_queue = queue.Queue()
            thread = threading.Thread(target=stream_subprocess_output, args=(process, output_queue))
            thread.start()
            
            analysis_result = None
            
            while True:
                try:
                    msg_type, content = output_queue.get(timeout=1)
                    
                    if msg_type == 'stdout':
                        # Check if this is the final JSON result
                        if content.startswith('{') and content.endswith('}'):
                            try:
                                analysis_result = json.loads(content)
                                yield f"data: [ANALYSIS] Analysis completed successfully\n\n"
                                break
                            except json.JSONDecodeError:
                                yield f"data: [ANALYSIS] {content}\n\n"
                        else:
                            yield f"data: [ANALYSIS] {content}\n\n"
                    
                    elif msg_type == 'stderr':
                        yield f"data: [WARNING] {content}\n\n"
                    
                    elif msg_type == 'done':
                        if content != 0:
                            yield f"data: [ERROR] Analysis failed with exit code {content}\n\n"
                            return
                        break
                    
                    elif msg_type == 'error':
                        yield f"data: [ERROR] Subprocess error: {content}\n\n"
                        return
                        
                except queue.Empty:
                    continue
            
            thread.join()
            
            # Generate Tavus video if analysis succeeded
            if analysis_result and analysis_result.get('status') == 'success':
                yield f"data: [TAVUS] Generating personalized coaching video...\n\n"
                
                try:
                    # Prepare Tavus API request
                    tavus_payload = {
                        "replica_id": REPLICA_ID,
                        "script": f"Hello! I've analyzed your javelin throw and found some areas for improvement. "
                                f"Your {analysis_result.get('most_deviant_angle', 'technique')} needs attention. "
                                f"{analysis_result.get('llm_suggestion', 'Keep practicing and focus on your form!')} "
                                f"Remember, small adjustments can lead to big improvements in your performance.",
                        "background": "office",
                        "variables": {
                            "athlete_name": "Athlete",
                            "main_issue": analysis_result.get('most_deviant_angle', 'technique'),
                            "suggestion": analysis_result.get('llm_suggestion', 'Keep practicing!')
                        }
                    }
                    
                    headers = {
                        "x-api-key": TAVUS_API_KEY,
                        "Content-Type": "application/json"
                    }
                    
                    response = requests.post(TAVUS_API_URL, json=tavus_payload, headers=headers)
                    
                    if response.status_code == 200:
                        tavus_result = response.json()
                        video_url = tavus_result.get('video_url', '')
                        
                        yield f"data: [TAVUS] Video generated successfully!\n\n"
                        yield f"data: {json.dumps({'tavus_video_url': video_url, 'analysis': analysis_result})}\n\n"
                    else:
                        yield f"data: [TAVUS] Failed to generate video: {response.text}\n\n"
                        yield f"data: {json.dumps({'analysis': analysis_result})}\n\n"
                        
                except Exception as e:
                    yield f"data: [TAVUS] Error generating video: {str(e)}\n\n"
                    yield f"data: {json.dumps({'analysis': analysis_result})}\n\n"
            else:
                yield f"data: [ERROR] Analysis did not complete successfully\n\n"
                
        except Exception as e:
            yield f"data: [ERROR] Unexpected error: {str(e)}\n\n"
        
        finally:
            # Cleanup
            if video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    yield f"data: [CLEANUP] Temporary file removed\n\n"
                except Exception as e:
                    yield f"data: [WARNING] Could not remove temporary file: {str(e)}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/plain',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*'
        }
    )

if __name__ == '__main__':
    print("Starting AI Javelin Coach server...")
    print(f"Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)