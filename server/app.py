from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
import os
import sys
import subprocess
from flask import send_file

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import simulator module
from server.modules.simulator import MujocoSimulator
from server.modules.openai_API import OpenAI_API
from server.modules.api_store import API_KEY

import time


# flask : https://www.runoob.com/flask/flask-views-functions.html
# socketio : https://python-socketio.readthedocs.io/en/latest/server.html

app = Flask(__name__, 
    template_folder='../templates',  # Templates are in the parent directory
    static_folder='../assets'         # Static files are in the assets directory
)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Create simulator instance
simulator = MujocoSimulator()

# Socket.IO event handlers
@socketio.on("connect")
def handle_connect(auth=None):  # Add auth parameter because Socket.IO passes it
    print("Client connected")
    # Send renderer info to client - ensure it's a string, not an object
    socketio.emit("renderer_info", {"renderer": simulator.best_renderer_name})

# LLM Part
@socketio.on("submitQuestion")
def llm_generation(data):
    question = data.get("question")
    api = OpenAI_API(API_KEY)  
    
    try:
        # Get response and blackboard words from the API
        llm_response, writing_letters = api.generate_response(question)
        
        # Send LLM response to client first
        socketio.emit("LLM_response", {"response": llm_response})
        
        
        socketio.emit("generating_sound", {"status": "starting"})
        PYTHON_ENV_PATH = "/opt/miniconda3/envs/tts/bin/python"
        SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules", "text_to_sound.py")
        print("SCRIPT_PATH:",SCRIPT_PATH)
        result = subprocess.run(
        [PYTHON_ENV_PATH, SCRIPT_PATH, llm_response],
        capture_output=True,
        text=True)
        
        socketio.emit("generating_sound", {"status": "completed"})
        
        if result.returncode == 0:
            print("TTS handle successful, sending socket message")
            print("Generating action data for output sound")
            PYTHON_ENV_PATH ='/opt/miniconda3/bin/python'
            SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "../outside_project/PantoMatrix/test_camn_audio.py"))
            print("SCRIPT_PATH:",SCRIPT_PATH)
            
            socketio.emit("generating_sound", {"status": "started_action"})
            
            result = subprocess.run(
                [PYTHON_ENV_PATH, SCRIPT_PATH],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(SCRIPT_PATH)) #When the main program calls a subroutine, the default is to use the current main program's working directory as the "relative path reference"
            
            
            
            if result.returncode == 0:
                socketio.emit("generating_sound", {"status": "completed_action"})
                print("Action generation successful, sending socket message")
            else :
                print("Action generation failed, sending error messages")
                socketio.emit("speech_error", {"status": "failed", "error": result.stderr})
            
        else:
            print("TTS failed, sending error messages")
            print(result.stderr)
            socketio.emit("speech_error", {"status": "failed", "error": result.stderr})
        
        
        # get maximum frame of the actions
        # max_frame =simulator.motion_data['poses'].shape[0]

        
        
        # Active 
        socketio.emit("generating_sound",{"status": "Started_teaching"})
        socketio.emit("word_to_speech", {"response": "output.wav"})
        simulator.execute_teaching_sequence(socketio.emit, writing_letters)

        
            
    except Exception as e:
        print(f"Error in LLM generation: {e}")
        socketio.emit("LLM_response", {"response": f"An error occurred: {str(e)}"})

@socketio.on('start_writing')
def handle_start_writing(data):
    """Start writing a letter"""
        # Start writing the first letter - ONLY if simulation is running
    if simulator.simulation_running and not simulator.writing_in_progress:
        first_letter = "Z"
        print(f"Starting writing simulation for letter: {first_letter}")
        simulator.writing_simulation(first_letter, socketio.emit)
        

@socketio.on('turning_around')
def handle_start_turning_around(data):
    if simulator.simulation_running and not simulator.writing_in_progress:
        result = simulator.turning_around_simulation(socketio.emit,[0,0,3.14/2])
        # Start writing the first letter - ONLY if simulation is running
        return result



@socketio.on('start_walking')
def handle_start_walking(data):
    """Start walking to a position"""
    if 'target_position' not in data:
        return {'success': False, 'message': 'Target position not provided'}
    
    target_position = data['target_position']
    
    # Validate target position
    if not isinstance(target_position, list) or len(target_position) < 2:
        return {'success': False, 'message': 'Invalid target position format'}
    
    # Ensure the simulator exists
    if simulator is None:
        return {'success': False, 'message': 'Simulator not initialized'}
    
    # Start walking
    try:
        result = simulator.walk_simulation(socketio.emit)
        # Start writing the first letter - ONLY if simulation is running
        return result
    except Exception as e:
        print(f"Error starting walking: {e}")
        return {'success': False, 'message': f'Error: {str(e)}'}
    
    

# WebSocket event (continuous)
@socketio.on("start_sim")
def start_sim():
    return simulator.start_simulation(socketio.emit)

@socketio.on("stop_sim")
def stop_sim():
    return simulator.stop_simulation()

# Flask HTTP Routes (one time request)
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/simulator")
def simulator_page():
    return render_template("simulator/index.html")



# Routes for Physical Control section
@app.route("/physical-control")
def physical_control_page():
    return render_template("physical-control/index.html")

@app.route("/physical-control/theory")
def physical_control_theory():
    return render_template("physical-control/theory.html")

@app.route("/physical-control/comparison")
def physical_control_comparison():
    return render_template("physical-control/comparison.html")

@app.route("/physical-control/visualization")
def physical_control_visualization():
    return render_template("physical-control/visualization.html")

@app.route("/physical-control/troubleshooting")
def physical_control_troubleshooting():
    return render_template("physical-control/troubleshooting.html")

@app.route("/physical-control/implementation")
def physical_control_implementation():
    return render_template("physical-control/implementation.html")


@app.route("/audio/output.wav")
def get_output_audio():                                
    return send_file("output.wav", mimetype="audio/wav") # same directory as app.py
       


if __name__ == "__main__":
    print("\n" + "="*80)
    print(f"MuJoCo 4K Simulation Server (using {simulator.best_renderer_name} renderer)")
    print("Visit http://127.0.0.1:6013 to view the simulation")
    print("=" * 80 + "\n")
    
    try:
        socketio.run(app, host="127.0.0.1", port=6013, debug=True, allow_unsafe_werkzeug=True)
    except Exception as e:
        print(f"Failed to start server: {e}")
        try:
            # Try using eventlet as alternative
            import eventlet
            eventlet.monkey_patch()
            socketio.run(app, host="127.0.0.1", port=6013, debug=True)
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            # Use Flask's built-in server
            app.run(host="127.0.0.1", port=6013, debug=True)

