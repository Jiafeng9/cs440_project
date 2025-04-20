/**
 * MuJoCo Simulator Module JavaScript
 * Handles WebSocket communication and UI interactions for the simulation
 */

// Performance statistics
let frameCount = 0;
let lastFrameTime = Date.now();
let clientFps = 0;

// Create Socket connection
const socket = io.connect(window.location.origin, {
    reconnection: true,
    reconnectionAttempts: 5,
    reconnectionDelay: 1000
});

// Safe DOM element selection
function getElement(selector) {
    return document.querySelector(selector);
}

// Debounce function to prevent multiple rapid submissions
function debounce(func, wait) {
    let timeout;
    return function() {
        const context = this, args = arguments;
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(context, args), wait);
    };
}



// DOM elements
const connectionStatus = getElement('#connection_status');
const rendererInfo = getElement('#renderer_info');
const serverFps = getElement('#server_fps');
const clientFps_elem = getElement('#client_fps');
const frameCount_elem = getElement('#frame_count');
const currentResolution = getElement('#current_resolution');
const simView = getElement('#sim_view');
const blackBoard_view = getElement('#chalk_trajectory');
const userQuestion = getElement('#user_question');
const llmResponseContainer = getElement('#LLM_response');
const submitBtn = getElement('#submitQuestionbtn');
const audio = document.getElementById("robotic_sound");
const blackBoardWord = document.getElementById("black_board_word");

// Create an error container if it doesn't exist
let errorContainer = getElement('#error_container');
if (!errorContainer) {
    errorContainer = document.createElement('div');
    errorContainer.id = 'error_container';
    errorContainer.className = 'alert alert-danger d-none mt-3';
    // Add it after the connection info card
    const connectionInfo = getElement('.connection-info');
    if (connectionInfo) {
        connectionInfo.after(errorContainer);
    } else {
        document.querySelector('.container').prepend(errorContainer);
    }
}

// Create a status message container
let statusContainer = getElement('#status_container');
if (!statusContainer) {
    statusContainer = document.createElement('div');
    statusContainer.id = 'status_container';
    statusContainer.className = 'alert alert-info d-none mt-3';
    if (errorContainer) {
        errorContainer.after(statusContainer);
    }
}

// Single handler for question submission with debounce
const handleSubmitQuestion = debounce(function() {
    if (userQuestion && userQuestion.value.trim()) {
        // Disable button during processing
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = 'Processing...';
        }
        
        // Show a loading indicator in the response area
        if (llmResponseContainer) {
            llmResponseContainer.innerHTML = '<div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div> Generating response...';
        }
        
        socket.emit('submitQuestion', {question: userQuestion.value});
        console.log('Submitting question:', userQuestion.value);
    }
}, 500); // 500ms debounce time

// Apply the debounced handler to both methods
if (submitBtn) {
    // Remove any existing listeners (safer approach)
    const newSubmitBtn = submitBtn.cloneNode(true);
    submitBtn.parentNode.replaceChild(newSubmitBtn, submitBtn);
    
    // Re-assign the reference and add new listener
    const freshSubmitBtn = getElement('#submitQuestionbtn');
    if (freshSubmitBtn) {
        freshSubmitBtn.addEventListener('click', handleSubmitQuestion);
    }
}

// Replace global function (for HTML onclick attribute)
window.submitQuestion = handleSubmitQuestion;

// Connection events
socket.on('connect', function() {
    console.log('Connection successful');
    if (connectionStatus) {
        connectionStatus.innerText = 'Connected';
        connectionStatus.className = 'status connected';
    }
    hideError();
});


//


function startWalking() {
    // Send socket event to start walking
    socket.emit('start_walking', {
        target_position: [0, 4.1, 0.8]  // Default target position
    }, function(response) {
        console.log('Walking response:', response);
        if (response && response.success) {
            alert('Robot started walking!');
        } else {
            alert('Failed to start walking: ' + (response ? response.message : 'Unknown error'));
        }
    });
}

function startWriting() {
    // Send socket event to start walking
    socket.emit('start_writing', {
    }, function(response) {
        console.log('Walking response:', response);
    });
}



function turnAround() {
    // Send socket event to start walking
    socket.emit('turning_around', {
    }, function(response) {
        console.log('Turning around response:', response);
    });
}


// Add this socket listener to receive walking status updates
socket.on('walking_status', function(data) {
    console.log('Walking status:', data);
    if (data.status === 'completed') {
        alert('Walking completed!');
    }
});


// LLM response handler
socket.on('LLM_response', function(data) {
    console.log('Received LLM response:', data.response);
    const raw = data.response;
    const formatted = formatLLMResponse(raw);
    if (llmResponseContainer) {
        llmResponseContainer.innerHTML = `<p>${formatted}</p>`;
    }
    // Re-enable the submit button
    if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.innerHTML = 'Submit';
    }
});

function formatLLMResponse(text) {
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')        // **加粗**
        .replace(/\*(.*?)\*/g, '<em>$1</em>')                    // *斜体*
        .replace(/^\d+\.\s+(.*)$/gm, '<li>$1</li>')              // 有序列表
        .replace(/\n{2,}/g, '</p><p>')                           // 段落换行
        .replace(/^([^-].*?):\s*$/gm, '<strong>$1:</strong>');   // 小标题
}


// Status updates for writing
socket.on('writing_status', function(data) {
    console.log('Writing status:', data);
    if (statusContainer) {
        let message = '';
        switch(data.status) {
            case 'starting':
                message = `Starting to write letter ${data.letter}...`;
                break;
            case 'started':
                message = `Writing letter ${data.letter}...`;
                break;
            case 'progress':
                message = `Writing letter ${data.letter} in progress...`;
                break;
            case 'completed':
                message = `Successfully wrote letter ${data.letter}!`;
                setTimeout(() => {
                    statusContainer.classList.add('d-none');
                }, 3000); // Hide after 3 seconds
                break;
            default:
                message = `Writing status: ${data.status}`;
        }
        statusContainer.textContent = message;
        statusContainer.classList.remove('d-none');
    }
});


socket.on('sound_with_movement_progress', function(data) {
    let message = `The percentage of the robot moving is ${data.progress}%`;
    if (statusContainer) {
        statusContainer.textContent = message;
        statusContainer.classList.remove('d-none');
    }
})


//status on generating sound
socket.on('generating_sound',function(data) {
    let message = '';
        switch(data.status) {
            case 'starting':
                message = `Starting to generating the video...`;
                break;
            case 'started_action':
                message =`Starting to generating actions for robot...`;
            case 'completed_action':
                message = `Finished generating actions for robot`;
                break;
            
            case 'completed':
                message = `Finished generating the video for given content!`;
                setTimeout(() => {
                    statusContainer.classList.add('d-none');
                }, 3000); // Hide after 3 seconds
                break;
            
            case 'Started_teaching':
                message =`Started teaching`;
            default:
                message = `Writing status: ${data.status}`;
        }
        statusContainer.textContent = message;
        statusContainer.classList.remove('d-none');
});


socket.on('disconnect', function() {
    console.log('Connection lost');
    if (connectionStatus) {
        connectionStatus.innerText = 'Disconnected';
        connectionStatus.className = 'status disconnected';
    }
    showError('Connection lost. Attempting to reconnect...');
});

socket.on('connect_error', function(error) {
    console.error('Connection error:', error);
    if (connectionStatus) {
        connectionStatus.innerText = 'Connection Error';
        connectionStatus.className = 'status disconnected';
    }
    showError('Connection error: ' + error);
});

// Receive renderer info
socket.on('renderer_info', function(data) {
    console.log('Using renderer:', data.renderer);
    if (rendererInfo) {
        rendererInfo.innerText = data.renderer;
    }
});

// Receive frame data
socket.on('frame', function(data) {
    // Update image
    if (simView) {
        simView.src = 'data:image/jpeg;base64,' + data.frame;
    }
    
    // Update performance statistics
    if (serverFps) serverFps.innerText = data.server_fps;
    if (frameCount_elem) frameCount_elem.innerText = data.frame_number;
    if (currentResolution) currentResolution.innerText = data.resolution || "Unknown";
    
    // Calculate client FPS
    const now = Date.now();
    const elapsed = now - lastFrameTime;
    lastFrameTime = now;
    
    if (elapsed > 0) {
        clientFps = Math.round(1000 / elapsed);
        if (clientFps_elem) clientFps_elem.innerText = clientFps;
    }
    
    frameCount++;
    
    // Hide error message (if any)
    hideError();
});

// Receive background frame data
socket.on('background_frame', function(data) {
    // Update image
    if (blackBoard_view) {
        blackBoard_view.src = 'data:image/jpeg;base64,' + data.frame;
    }
});

// Receive error messages
socket.on('simulation_error', function(data) {
    showError('Simulation error: ' + data.error);
});

// Simulation status messages
socket.on('simulation_status', function(data) {
    if (statusContainer) {
        statusContainer.textContent = data.message || data.status;
        statusContainer.classList.remove('d-none');
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            statusContainer.classList.add('d-none');
        }, 5000);
    }
});

// Show error message
function showError(message) {
    console.error('Error:', message);
    if (errorContainer) {
        errorContainer.textContent = message;
        errorContainer.classList.remove('d-none');
    }
}

// Hide error message
function hideError() {
    if (errorContainer) {
        errorContainer.classList.add('d-none');
    }
}

// Control functions with debouncing
const startSimulation = debounce(function() {
    socket.emit('start_sim');
    console.log('Requesting simulation start');
    
    if (statusContainer) {
        statusContainer.textContent = 'Starting simulation...';
        statusContainer.classList.remove('d-none');
    }
}, 500);



const stopSimulation = debounce(function() {
    socket.emit('stop_sim');
    console.log('Requesting simulation stop');
    
    if (statusContainer) {
        statusContainer.textContent = 'Stopping simulation...';
        statusContainer.classList.remove('d-none');
    }
}, 500);



// Expose debounced functions globally
window.startSimulation = startSimulation;
window.stopSimulation = stopSimulation;


// write-to-speech
socket.on('word_to_speech', function(data) {
    audio.src = "/audio/output.wav";  // define the audio position
    audio.load();  // ensure load the new audio
    audio.play();  // play the audio
});




// Initialize the UI
document.addEventListener('DOMContentLoaded', function() {
    console.log('Page loaded, initializing simulator UI');
    
    // Set connection status initially to "Connecting"
    if (connectionStatus) {
        connectionStatus.innerText = 'Connecting...';
        connectionStatus.className = 'status connecting';
    }
    
    // Display placeholder for simulation image if needed
    if (simView && !simView.src) {
        simView.src = '/img/simulator-placeholder.jpg';
    }
});