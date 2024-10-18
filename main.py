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

        angle = np.arccos((np.dot(ba, bc)) / (np.linalg.norm(ba) * np.linalg.norm(bc))) * (180 / np.pi)

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

if __name__ == "__main__":
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Input number of users and their names
    num_persons = int(input("Enter the number of persons: ").strip())
    names = []
    for i in range(num_persons):
        name = input(f"Enter name of person {i+1}: ").strip()
        names.append(name)
    
    # Input the number of squats to perform
    total_squats = int(input("Enter the number of squats to perform: ").strip())

    results = []  # To store each person's name and time taken

    cap = None
    decision = int(input("Video(0) or Camera(1) ? : ").strip())
    if decision == 0:
        cap = cv2.VideoCapture("video.mp4")
    else:
        cap = cv2.VideoCapture(0)

    while cap.read()[1] is None:
        print("Waiting for Video")

    for person in names:
        print(f"Starting for {person}...")

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
                frame = cv2.resize(frame, (1024, 600))

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False

                lm = pose.process(frame_rgb).pose_landmarks
                lm_arr = lm.landmark if lm else []

                frame_rgb.flags.writeable = True
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(frame, lm, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(
                    0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

                if lm:
                    rAngle = findAngle(lm_arr[24], lm_arr[26], lm_arr[28])
                    lAngle = findAngle(lm_arr[23], lm_arr[25], lm_arr[27])

                    rState = legState(rAngle)
                    lState = legState(lAngle)
                    state = rState * lState

                    # Final state is product of two leg states
                    if state == 1 or state == 9:  # Upright or squatting
                        if lastState != state:
                            lastState = state
                            if lastState == 1:  # Squat position
                                repCount += 1
                                message = f"Rep completed!"
                                wave_obj.play()

                            # Stop when the person completes the required reps
                            if repCount >= total_squats:
                                end_time = time.time()  # End timer
                                time_taken = end_time - start_time
                                results.append((person, time_taken))
                                print(f"{person} completed in {time_taken:.2f} seconds.")
                                break

                # Display rep count on the frame
                cv2.putText(frame, f"Reps: {repCount}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 10)

                # Display message on the frame
                cv2.putText(frame, message, (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

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
    else:
        print(f"\nWinner: {results[0][0]}")

    cap.release()
    cv2.destroyAllWindows()
