import cv2
import mediapipe as mp
import numpy as np
import simpleaudio as sa
import time
import sys
import os

if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".")

wave_path = os.path.join(base_path, "ding.wav")
wave_obj = sa.WaveObject.from_wave_file(wave_path)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle

def display_leaderboard(results):
    leaderboard_img = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.putText(leaderboard_img, "Leaderboard", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    for i, (name, successful_reps, _) in enumerate(results):
        rank = i + 1
        cv2.putText(leaderboard_img, f"{rank}. {name} - {successful_reps} successful reps",
                    (30, 100 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 2)

    cv2.putText(leaderboard_img, "Press Escape to exit", (30, 550),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    cv2.imshow("Leaderboard", leaderboard_img)
    while True:
        if cv2.waitKey(10) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == 27:
            break

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

participants = int(input("Enter number of participants: "))
players = []
for i in range(participants):
    name = input(f"Enter name for participant {i + 1}: ")
    players.append(name)

time_limit = int(input("Enter the time limit in seconds: "))

results = []
cap = cv2.VideoCapture(0)

for player in players:
    print(f"{player}'s turn! Press 'q' or 'ESC' to stop.")
    successful_counter = 0
    unsuccessful_counter = 0
    total_counter = 0
    stage = "UP"  # Start in UP position
    squat_phase = "READY"  # Track squat progression
    reached_bottom = False  # Track if squat reached proper depth
    start_time = time.time()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1700, 900))

            current_time = time.time()
            elapsed_time = current_time - start_time
            remaining_time = max(0, time_limit - elapsed_time)

            if remaining_time <= 0:
                print(f"Time's up for {player}!")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results_pose = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results_pose.pose_landmarks.landmark

                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                angle_knee = calculate_angle(hip, knee, ankle)

                cv2.putText(image, str(angle_knee),
                            tuple(np.multiply(knee, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (79, 121, 66), 2, cv2.LINE_AA)

                # Improved squat detection logic
                if stage == "UP" and angle_knee > 169:
                    squat_phase = "READY"
                    reached_bottom = False
                elif stage == "UP" and angle_knee <= 169:
                    stage = "DOWN"
                    squat_phase = "DESCENDING"
                elif stage == "DOWN" and angle_knee <= 90:
                    reached_bottom = True
                    squat_phase = "BOTTOM"
                elif stage == "DOWN" and angle_knee > 169:
                    stage = "UP"
                    total_counter += 1
                    if reached_bottom:
                        successful_counter += 1
                        wave_obj.play()
                        squat_phase = "SUCCESSFUL"
                    else:
                        unsuccessful_counter += 1
                        squat_phase = "INCOMPLETE"

                # Update squat status display
                if angle_knee > 169:
                    squat_status = "Stand Straight"
                elif angle_knee <= 90:
                    squat_status = "Deep Enough"
                else:
                    squat_status = "Go Lower"

                # Display squat phase and status
                cv2.putText(image, f"Phase: {squat_phase}", (10, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(image, squat_status, (10, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

            except:
                pass

            # Display rectangle
            cv2.rectangle(image, (0, 0), (600, 150), (245, 117, 16), -1)

            # Display rep counts
            cv2.putText(image, 'SUCCESSFUL', (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(successful_counter),
                        (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, 'UNSUCCESSFUL', (200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(unsuccessful_counter),
                        (250, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, 'TOTAL', (450, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(total_counter),
                        (450, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Display remaining time
            cv2.putText(image, f"Time Left: {int(remaining_time)}s", (10, 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

            cv2.putText(image, f"Player: {player}", (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 75, 59), 5, cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == 27:
                break

    results.append((player, successful_counter, unsuccessful_counter))

cap.release()
cv2.destroyAllWindows()

# Sort results by successful reps (highest to lowest)
results.sort(key=lambda x: x[1], reverse=True)

display_leaderboard(results)

cap.release()
cv2.destroyAllWindows()