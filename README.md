# Squat Rep Counter using MediaPipe

This project is a **Squat Rep Counter** that uses the **MediaPipe Pose** solution for detecting body landmarks to count squats in real-time. The system processes video input from a webcam or video file, tracks key body landmarks, and counts the number of squats performed based on leg angle changes. The project also provides visual and audio feedback for each successful squat.

## Features

- **Pose Detection**: Utilizes the MediaPipe Pose model to detect key body landmarks.
- **Real-Time Squat Counting**: Detects and counts squats in real-time using a video feed or webcam.
- **State-Based Leg Tracking**: Determines leg state (squat, transition, or upright) using calculated joint angles.
- **Audio Feedback**: Plays a sound each time a full squat is detected.
- **Visual Feedback**: Displays pose landmarks, rep count, and prompts for correcting posture on the screen.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Srustika-18/squat-counter.git
   cd squat-counter
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r req.txt
   ```

   Required packages include:
   - `opencv-python`
   - `mediapipe`
   - `numpy`
   - `simpleaudio`

## Usage

1. **Run the Squat Rep Counter**:
   ```bash
   python main.py
   ```

2. **Select Input Source**:
   - Input `0` for a video file (`video.mp4`).
   - Input `1` for the webcam feed.

3. **Perform Squats**:
   - Ensure your legs are visible in the frame.
   - The system will track and count your squats. Each time you complete a full squat, it will play a "ding" sound and increment the rep counter.

### Keyboard Controls

- Press `Esc` to exit the video window.

## How It Works

- The system uses **MediaPipe** to detect body landmarks, focusing on the hip, knee, and ankle joints.
- The knee angle is calculated to determine leg states:
  - **Squat**: Knee angle < 90°.
  - **Transition**: Knee angle between 90° and 120°.
  - **Upright**: Knee angle > 120°.
- A full squat is counted when the system detects a transition from upright to squat and back to upright. Audio feedback is provided when a rep is successfully completed.

## Project Structure

```bash
.
├── main.py		# Main script for squat counting
├── req.txt		# Project dependencies
├── video.mp4		# Sample video (optional)
├── ding.wav		# Sound file for feedback
└── README.md		# Project documentation
```

## Dependencies

- **OpenCV**: For capturing video and processing frames.
- **MediaPipe**: For pose detection.
- **NumPy**: For calculating angles between joints.
- **SimpleAudio**: For playing sound when a rep is completed.

## Notes

- Ensure your legs are clearly visible in the camera frame for accurate detection.
- The system works best in a well-lit environment.
- The provided sound file (`ding.wav`) can be replaced with any other `.wav` file for custom feedback.

## Future Improvements

- Add additional exercise detection (e.g., lunges, push-ups).
- Improve pose correction prompts for better squat form.
- Implement a graphical user interface (GUI) to enhance usability and provide more detailed workout stats.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
