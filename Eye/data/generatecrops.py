from croprepeatingedge import crop_repeating_edge
import cv2
from loadsubject import load_subject
import math
import random
import os
import os.path

BASE_DIRECTORY = './raw'
OUTPUT_DIR = './processed'
REPLACE_EXISTING = False  # Whether to replace a processed directory if it already exists.


def generate_crops():
    subject_dirs = os.listdir(BASE_DIRECTORY)
    print("Subject dirs: ", subject_dirs)
    for current_subject in subject_dirs:
        print('Processing subject %s ...' % current_subject)
        subject_dir = os.path.join(BASE_DIRECTORY, current_subject)
        s = load_subject(subject_dir)

        if not REPLACE_EXISTING and os.path.exists(os.path.join(OUTPUT_DIR, current_subject)):
            # Don't replace the directory if it already exists.
            continue

        apple_face_dir = os.path.join(OUTPUT_DIR, current_subject, 'appleFace')
        apple_left_eye_dir = os.path.join(OUTPUT_DIR, current_subject, 'appleLeftEye')
        apple_right_eye_dir = os.path.join(OUTPUT_DIR, current_subject, 'appleRightEye')
        rect_dir = os.path.join(OUTPUT_DIR, current_subject, 'rect')

        os.makedirs(apple_face_dir, exist_ok=True)
        os.makedirs(apple_left_eye_dir, exist_ok=True)
        os.makedirs(apple_right_eye_dir, exist_ok=True)
        os.makedirs(rect_dir, exist_ok=True)

        for i in range(len(s['frames'])):
            frame_filename = s['frames'][i]
            frame = cv2.imread(os.path.join(subject_dir, 'frames', frame_filename))

            if math.isnan(s['appleFace']['x'][i]) or math.isnan(s['appleLeftEye']['x'][i])\
                    or math.isnan(s['appleRightEye']['x'][i]):
                continue

            while True:
                try:
                    face_image = crop_repeating_edge(frame, \
                            (s['appleFace']['x'][i], s['appleFace']['y'][i],\
                            s['appleFace']['w'][i], s['appleFace']['h'][i]))
                    # Random ints are designed to shift the eye detections a little to see if this makes a difference.
                    left_eye_image = crop_repeating_edge(face_image, \
                            (s['appleLeftEye']['x'][i] + random.randint(-10, 10), s['appleLeftEye']['y'][i] + random.randint(-10, 10),\
                            s['appleLeftEye']['w'][i] + random.randint(-5, 5), s['appleLeftEye']['h'][i] + random.randint(-5, 5)))
                    right_eye_image = crop_repeating_edge(face_image, \
                            (s['appleRightEye']['x'][i] + random.randint(-10, 10), s['appleRightEye']['y'][i] + random.randint(-10, 10),\
                            s['appleRightEye']['w'][i] + random.randint(-5, 5), s['appleRightEye']['h'][i] + random.randint(-5, 5)))
                    break
                except Exception:
                    print("Exception...continuing")
                    pass
            cv2.imwrite(os.path.join(apple_face_dir, frame_filename), face_image)
            cv2.imwrite(os.path.join(apple_left_eye_dir, frame_filename), left_eye_image)
            cv2.imwrite(os.path.join(apple_right_eye_dir, frame_filename), right_eye_image)

            fx, fy, fw, fh =\
                    (s['appleFace']['x'][i], s['appleFace']['y'][i],\
                    s['appleFace']['w'][i], s['appleFace']['h'][i])
            fx, fy, fw, fh = [int(round(c)) for c in (fx, fy, fw, fh)]
            face_im = frame[fy:fy+fh, fx:fx+fw, :]
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)
            ex, ey, ew, eh =\
                    (s['appleLeftEye']['x'][i], s['appleLeftEye']['y'][i],\
                    s['appleLeftEye']['w'][i], s['appleLeftEye']['h'][i])
            ex, ey, ew, eh = [int(round(c)) for c in (ex, ey, ew, eh)]
            ex, ey = fx+ex, fy+ey
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            ex, ey, ew, eh =\
                    (s['appleRightEye']['x'][i], s['appleRightEye']['y'][i],\
                    s['appleRightEye']['w'][i], s['appleRightEye']['h'][i])
            ex, ey, ew, eh = [int(round(c)) for c in (ex, ey, ew, eh)]
            ex, ey = fx+ex, fy+ey
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
            cv2.imwrite(os.path.join(rect_dir, frame_filename), frame)


if __name__ == '__main__':
    generate_crops()
