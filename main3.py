import cv2
import mediapipe as mp
import numpy as np
import simpleaudio as sa
import time

wave_obj = sa.WaveObject.from_wave_file("ding.wav")


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle


def rescale_frame(frame, percent=50):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def display_leaderboard(results):
    # Create a black image for the leaderboard
    leaderboard_img = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.putText(leaderboard_img, "Leaderboard", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    for i, (name, time_taken) in enumerate(results):
        rank = i + 1
        cv2.putText(leaderboard_img, f"{rank}. {name} - {time_taken:.2f} seconds",
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

target_reps = int(input("Enter the target number of reps: "))

results = []
cap = cv2.VideoCapture(0)

for player in players:
    print(f"{player}'s turn! Press 'q' or 'ESC' to stop.")
    counter = 0
    stage = None
    start_time = time.time()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1700, 900))

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results_pose = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results_pose.pose_landmarks.landmark

                # Get coordinates
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Calculate angle
                angle_knee = calculate_angle(hip, knee, ankle)

                cv2.putText(image, str(angle_knee),
                            tuple(np.multiply(knee, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (
                                79, 121, 66), 2, cv2.LINE_AA
                            )

                # Curl counter logic
                if angle_knee > 169:
                    stage = "UP"
                if angle_knee <= 90 and stage == 'UP':
                    stage = "DOWN"
                    counter += 1
                    wave_obj.play()
                    print(f"Reps: {counter}")

                if counter >= target_reps:
                    print(f"{player} completed the target!")
                    break

            except:
                pass

            # Render squat counter
            cv2.rectangle(image, (0, 0), (400, 73), (245, 117, 16), -1)

            # Rep data
            cv2.putText(image, 'REPS', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Stage data
            cv2.putText(image, 'STAGE', (65, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage,
                        (80, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, f"Player: {player}", (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 75, 59), 5, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(
                                          color=(245, 66, 230), thickness=2, circle_radius=2))

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == 27:
                break

    end_time = time.time()
    elapsed_time = end_time - start_time
    results.append((player, elapsed_time))

cap.release()
cv2.destroyAllWindows()

# Sort results by time
results.sort(key=lambda x: x[1])

display_leaderboard(results)

cap.release()
cv2.destroyAllWindows()
