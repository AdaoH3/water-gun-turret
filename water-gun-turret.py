
# WATER GUN TURRET (or whatever you want to hook up)
# This script uses a webcam stream (RTMP) w Mediapipe for pose detection, 2 servos for aiming (x and y), and a relay for firing

# make sure to install dependencies:
#pip install opencv-python mediapipe pigpio numpy 
#also make sure pigpiod is running on your raspberry pi (sudo pigpiod) and adjust RTMP_URL to your setup (e.g. nginx-rtmp)
#change the trigger pins dependig if your relay is active HIGH or LOW (mine is active LOW, so 0 = fire, 1 = idle)
# :)

import cv2
import mediapipe as mp
import pigpio
import numpy as np
import time
import threading
import sys
import datetime

#config lines
#local ip address of machine running RTMP server (e.g. nginx-rtmp)
RTMP_URL = "rtmp://192.168.1.81/live"

#set the hour and miute
START_HOUR = 4
START_MIN = 25

SERVO_PIN_X = 14
SERVO_PIN_Y = 15
TRIGGER_PIN = 18

#calibrated pulse widths for servos (range is 500-2500, 1500 is center -> adjust as needed))
X_MIN = 700
X_MAX = 2300
Y_MIN = 1450
Y_MAX = 2000

SMOOTHING = 0.2
MAX_STEP = 25
DEADZONE = 20

#fire contrrol (duration -> how log the trigger stays LOW, interval -> max time betwee fires (WHEN PERSON DETECTED))
FIRE_DURATION = 2.0      
FIRE_INTERVAL = 20.0   

#=======================================Camera Stream Class=======================================
#Uses threading to read framed from RTMP without being hella laggy (too much buffering causes WAY too much lag)
class CameraStream:
def __init__(self, url):
    self.cap = cv2.VideoCapture(url)
    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    self.frame = None
    self.running = True
    self.lock = threading.Lock()
    threading.Thread(target=self.update, daemon=True).start()

def update(self):
    while self.running:
        self.cap.grab()
        ret, frame = self.cap.retrieve()
        if ret:
            with self.lock:
                self.frame = frame

def read(self):
    with self.lock:
        return self.frame.copy() if self.frame is not None else None

def stop(self):
    self.running = False
    self.cap.release()

#===============================MediaPipe Setup (Perso and Face Detection Model)=======================================

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
model_complexity=0, #lightweight model
min_detection_confidence=0.6, #applies to whole person detection (when first time detect)
min_tracking_confidence=0.6, #tracks landmarks after initial detection [tracks off of last pose, faster]
smooth_landmarks=True #more stable motion
)

#===============================GPIO Setup [I always use pigpio w raspberry pi]=======================================

pi = pigpio.pi()
if not pi.connected:
print("pigpiod not initalized")
exit()

#center servos
current_pulse_x = 1500.0
current_pulse_y = 1600.0

pi.set_servo_pulsewidth(SERVO_PIN_X, int(current_pulse_x))
pi.set_servo_pulsewidth(SERVO_PIN_Y, int(current_pulse_y))

# IMPORTANT: my trigger relay is active LOW (0 = fire, 1 = idle) — adjust if your setup is different
pi.set_mode(TRIGGER_PIN, pigpio.OUTPUT)
pi.write(TRIGGER_PIN, 1)

#=============================tracking state======================================

smoothed_x = None
smoothed_y = None
last_center_x = None
last_center_y = None

frame_count = 0
DETECT_EVERY_N = 3

#fire timing
last_fire_time = time.time() - FIRE_INTERVAL
firing = False
fire_start_time = 0

#=============================stream setup======================================

cam = CameraStream(RTMP_URL)
print("waiting for stream...")
while cam.read() is None:
time.sleep(0.05)

cv2.namedWindow("Face Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Tracking", 848, 480)

#=============================main loop======================================

#start time clock, script not started until time set reached 
def wait_until_start(hour, minute):
    start_time = datetime.time(hour, minute)
    print(f"Waiting for start time {start_time}...")

    while True:
        now = datetime.datetime.now().time()

        if now >= start_time:
            print("starting")
            return

        time.sleep(5)

wait_until_start(START_HOUR, START_MIN) 

#open cv stuffs
try:
    while True:
        frame = cam.read()
        if frame is None:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        center_x = w // 2
        center_y = h // 2

        cv2.drawMarker(frame, (center_x, center_y),
                    (255, 0, 0), cv2.MARKER_CROSS, 20, 2)

        frame_count += 1

        #mediapipe pose detection every N frames (to not be too intesive on CPU)
        person_detected = False

        if frame_count % DETECT_EVERY_N == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                nose = landmarks[mp_pose.PoseLandmark.NOSE] #finds the nose off of pose detection (body)

                if nose.visibility > 0.6:
                    last_center_x = int(nose.x * w)
                    last_center_y = int(nose.y * h)
                    person_detected = True
                else:
                    last_center_x = None
                    last_center_y = None
            else:
                last_center_x = None
                last_center_y = None

#====------------------ servo control ----------------------
        if last_center_x is not None:
            if smoothed_x is None:
                smoothed_x = float(last_center_x)
                smoothed_y = float(last_center_y)
            else:
                #helps turret not jitter like crazy, adjust SMOOTHING for more/less smoothing
                smoothed_x += SMOOTHING * (last_center_x - smoothed_x) 
                smoothed_y += SMOOTHING * (last_center_y - smoothed_y)

            error_x = smoothed_x - center_x
            error_y = smoothed_y - center_y

            #calculates error (to account for deadzone space, turret can only be so accurate)
            # and then pixel to pulse width conversion
            if abs(error_x) > DEADZONE or abs(error_y) > DEADZONE:
                #maps smoothed pixel to pmw range for servo
                target_x = np.interp(smoothed_x, [0, w], [X_MIN, X_MAX])
                target_y = np.interp(smoothed_y, [0, h], [Y_MIN, Y_MAX])

                #breaks down movement into steps to avoid overshooting
                delta_x = np.clip(target_x - current_pulse_x, -MAX_STEP, MAX_STEP)
                delta_y = np.clip(target_y - current_pulse_y, -MAX_STEP, MAX_STEP)

                current_pulse_x += delta_x
                current_pulse_y += delta_y

                #clamps at servo limits
                current_pulse_x = np.clip(current_pulse_x, X_MIN, X_MAX)
                current_pulse_y = np.clip(current_pulse_y, Y_MIN, Y_MAX)

                #sets signal to servos
                pi.set_servo_pulsewidth(SERVO_PIN_X, int(current_pulse_x))
                pi.set_servo_pulsewidth(SERVO_PIN_Y, int(current_pulse_y))

# ---------------------- trigger control ----------------------

        current_time = time.time()

        if person_detected and not firing and (current_time - last_fire_time >= FIRE_INTERVAL):
            #SWITCH TO 1 if your trigger active HIGH (most relays are active LOW tho)
            pi.write(TRIGGER_PIN, 0)
            firing = True
            fire_start_time = current_time

        if firing and (current_time - fire_start_time >= FIRE_DURATION):
            pi.write(TRIGGER_PIN, 1)
            firing = False
            last_fire_time = current_time

        #display status for person detected and cooldown state
        status = f"Detected:{person_detected} Cooldown:{int(current_time-last_fire_time)}"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Face Tracking", frame)

        #press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cam.stop()
    pose.close()
    pi.write(TRIGGER_PIN, 1) #switch off trigger
    pi.set_servo_pulsewidth(SERVO_PIN_X, 0)
    pi.set_servo_pulsewidth(SERVO_PIN_Y, 0)
    pi.stop()
    cv2.destroyAllWindows()