# RoboLab: Robotics Education & Research Platform

RoboLab is a comprehensive web-based platform for robotics education and research. It features a high-performance MuJoCo simulator, research tracking tools, and visualization components.

## Features

- **High-Performance MuJoCo Simulator**: Real-time physics simulation with adjustable quality settings
- **Research Tracker**: Keep up with the latest papers and advances in robotics
- **Visualization Tools**: Analyze motion tracking and trajectory optimization

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/robolab.git
   cd robolab
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Prepare the MuJoCo model files:
   - Place your `.xml` model files in the `models/` directory
   - By default, the application will look for `classroom.xml`

## Usage

1. Start the server:
   ```
   python server/app.py
   ```

2. Open your browser and go to:
   ```
   http://127.0.0.1:6006
   ```

## Simulator Configuration

The simulator supports various quality and performance settings:

- **Resolution**: From 320x240 to 4K (2560x1920)
- **Image Quality**: Adjustable JPEG compression 
- **Rendering Frequency**: Control frame rendering rate
- **Physics Accuracy**: Adjust physics substeps for accuracy vs. performance
- **Shadows**: Enable/disable for visual quality vs. performance

## Project Structure

```
robolab/
├── assets/                     # Static assets (CSS, JS, images)
├── models/                     # MuJoCo model files
├── server/                     # Server-side code
│   ├── app.py                  # Main Flask application
│   ├── config.py               # Configuration file
│   └── modules/                # Server modules
│       └── simulator.py        # Simulation module
├── templates/                  # HTML templates
│   ├── index.html              # Main page
│   └── simulator/              # Simulator templates
└── requirements.txt            # Python dependencies
```

## Renderer Support

The application automatically detects and uses the best available renderer:

- **GLFW**: Default for most platforms, requires a display
- **EGL**: Hardware-accelerated rendering without a display (Linux)
- **OSMesa**: Software-based rendering, slower but works everywhere

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MuJoCo physics engine - https://mujoco.org/
- Flask web framework - https://flask.palletsprojects.com/
- Socket.IO for real-time communication - https://socket.io/