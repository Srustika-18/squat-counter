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

    num_people = int(input("Enter the number of people: ").strip())
    people = []
    for i in range(num_people):
        name = input(f"Enter name for person {i+1}: ").strip()
        people.append({'name': name, 'reps': 0, 'time': 0})

    num_squats = int(input("Enter the number of squats to complete: ").strip())

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        person_idx = 0
        lastState = 9  # Start in upright position
        message = ""

        start_time = time.time()
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

                # Debug information
                cv2.putText(frame, f"R Angle: {rAngle:.2f}", (10, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                cv2.putText(frame, f"L Angle: {lAngle:.2f}", (10, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                cv2.putText(frame, f"R State: {rState}", (10, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(frame, f"L State: {lState}", (10, 310),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                # Squat logic
                if state == 0:  # One or both legs not detected
                    message = "Legs not fully detected"
                elif state % 2 == 0 or rState != lState:  # Transitioning
                    if lastState == 1:
                        message = "Fully extend legs"
                    else:
                        message = "Fully retract legs"
                else:
                    if state == 1 or state == 9:
                        if lastState != state:
                            lastState = state
                            if lastState == 1:
                                people[person_idx]['reps'] += 1
                                message = f"Rep completed for {people[person_idx]['name']}!"
                                wave_obj.play()

                                if people[person_idx]['reps'] >= num_squats:
                                    people[person_idx]['time'] = time.time() - start_time
                                    print(f"{people[person_idx]['name']} completed {num_squats} squats in {people[person_idx]['time']:.2f} seconds!")

                                    # Switch to next person
                                    person_idx += 1
                                    if person_idx >= num_people:
                                        print("All participants have completed their squats.")
                                        break
                                    else:
                                        start_time = time.time()  # Reset time for the next person

            # Display rep count on the frame
            cv2.putText(frame, f"Reps: {people[person_idx]['reps']}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 10)

            # Display message on the frame
            cv2.putText(frame, message, (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

            cv2.imshow("Squat Rep Counter", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

    # Final results for all participants
    print("\nFinal Results:")
    for person in people:
        print(f"{person['name']} completed {num_squats} squats in {person['time']:.2f} seconds.")
