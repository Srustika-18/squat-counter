import sys
import mediapipe as mp
import cv2
import numpy as np
import simpleaudio as sa
import time

wave_obj = sa.WaveObject.from_wave_file("ding.wav")


def findAngle(a, b, c, minVis=0.8):
    if a.visibility > minVis and b.visibility > minVis and c.visibility > minVis:
        bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])

        angle = np.arccos((np.dot(ba, bc)) / (np.linalg.norm(ba)
                          * np.linalg.norm(bc))) * (180 / np.pi)

        if angle > 180:
            return 360 - angle
        else:
            return angle
    else:
        return -1


def legState(angle):
    if angle < 0:
        return 0  # Joint is not being picked up
    elif angle < 90:
        return 1  # Squat range
    elif angle < 120:
        return 2  # Transition range
    else:
        return 3  # Upright range


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
        if cv2.waitKey(1) & 0xFF == 27:  # Wait for 'Escape' key
            break

# Angle Visualizer


def draw_angle_indicator(frame, angle, position):
    """
    Draws a visual indicator for the angle on the frame.
    """
    x, y = position
    cv2.rectangle(frame, (x, y), (x + 300, y + 50),
                  (50, 50, 50), -1)  # Background bar
    cv2.rectangle(frame, (x, y), (x + int((angle / 180) * 300),
                  y + 50), (0, 255, 0), -1)  # Fill bar
    cv2.putText(frame, f"Angle: {int(angle)} deg", (x + 10, y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


# Updated feedback logic with text guidance
def feedback_on_squat(state, lastState, lState, rState):
    """
    Returns feedback text based on the squat state.
    """
    if state == 0:  # Joint not detected
        return "Legs not fully detected"
    elif state % 2 == 0 or rState != lState:  # Transitioning state
        if lastState == 1:
            return "Fully extend legs" if rState == 2 or lState == 2 else "Correct your posture"
        else:
            return "Fully retract legs" if rState == 3 or lState == 3 else "Adjust your squat depth"
    else:
        return "Good form!" if state in {1, 9} else ""


if __name__ == "__main__":
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = None
    decision = int(input("Video(0) or Camera(1) ? : ").strip())
    if decision == 0:
        cap = cv2.VideoCapture("video.mp4")
    else:
        cap = cv2.VideoCapture(0)

    while cap.read()[1] is None:
        print("Waiting for Video")

    results = []  # To store each person's name and time taken

    # Input number of users and their names
    num_persons = int(input("Enter the number of persons: ").strip())
    names = []
    for i in range(num_persons):
        name = input(f"Enter name of person {i + 1}: ").strip()
        names.append(name)

    # Input the number of squats to perform
    total_squats = int(
        input("Enter the number of squats to perform: ").strip())

    for person in names:
        print(f"Starting for {person}...")
        time.sleep(5)

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            repCount = 0
            lastState = 9  # Start in upright position
            message = ""
            start_time = time.time()  # Start timer for the person

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    print('Error: Image not found or could not be loaded.')
                    break
                frame = cv2.resize(frame, (1920, 1080))

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False

                lm = pose.process(frame_rgb).pose_landmarks
                lm_arr = lm.landmark if lm else []

                frame_rgb.flags.writeable = True
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(frame, lm, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(
                                              color=(0, 255, 0), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

                if lm:
                    rAngle = findAngle(lm_arr[24], lm_arr[26], lm_arr[28])
                    lAngle = findAngle(lm_arr[23], lm_arr[25], lm_arr[27])

                    rState = legState(rAngle)
                    lState = legState(lAngle)
                    state = rState * lState

                    # Visual Angle Indicator
                    draw_angle_indicator(frame, rAngle, (50, 500))
                    draw_angle_indicator(frame, lAngle, (450, 500))

                    # Feedback text
                    message = feedback_on_squat(
                        state, lastState, lState, rState)

                    # Rep counting logic
                    if state in {1, 9} and lastState != state:
                        lastState = state
                        if lastState == 1:  # Squat position
                            repCount += 1
                            wave_obj.play()
                            message = "Rep completed!"

                    # End condition
                    if repCount >= total_squats:
                        end_time = time.time()
                        time_taken = end_time - start_time
                        results.append((person, time_taken))
                        print(f"{person} completed in {time_taken:.2f} seconds.")
                        break

                # Display UI elements
                cv2.putText(frame, f"Reps: {repCount}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 10)
                cv2.putText(frame, message, (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                cv2.putText(frame, f"Player: {person}", (10, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 75, 59), 5)
                cv2.imshow(f"Squat Rep Counter - {person}", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    # Rank the users based on the time taken
    results.sort(key=lambda x: x[1])  # Sort by time taken (ascending)

    # Display the ranking
    print("\nRanking:")
    for i, (name, time_taken) in enumerate(results):
        rank = i + 1
        print(f"{rank}. {name} - {time_taken:.2f} seconds")

    # Determine the winner and top 3
    if len(results) >= 3:
        print(f"\nWinner: {results[0][0]}")
        print(f"2nd Place: {results[1][0]}")
        print(f"3rd Place: {results[2][0]}")
    elif len(results) == 2:
        print(f"\nWinner: {results[0][0]}")
        print(f"2nd Place: {results[1][0]}")
    elif len(results) == 1:
        print(f"\nWinner: {results[0][0]}")

    # Display leaderboard
    display_leaderboard(results)

    cap.release()
    cv2.destroyAllWindows()
