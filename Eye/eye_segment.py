import cv2
import numpy as np


class EyeExtractor:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    MAX_EYE_PCT_DIST = 0.25  # Eyes can't be more than 0.25 * eye_bb_h from other eyes.

    @staticmethod
    def __extract_roi(img, roi_rect):
        x, y, w, h = roi_rect
        return img[y:y+h, x:x+w]

    @staticmethod
    def __get_best_two_eyes(face_eyes):
        '''
        Returns the best two eyes for the given face and eyes.
        '''
        face, eyes = face_eyes

        # Discard eye candidates which do not have a pair at a similar y-coordinate.
        eyes.sort(key=lambda eye: eye[1])
        nearby_eyes_per_eye = []  # A list of nearby eyes for each eye
        for i, eye in enumerate(eyes):
            max_dist = eye[3] * EyeExtractor.MAX_EYE_PCT_DIST
            nearby_eyes = [eye]
            for e in eyes[i + 1:]:
                if e[1] < eye[1] + max_dist:
                    nearby_eyes.append(e)
                else:
                    break
            for e in reversed(eyes[:i]):
                if e[1] > eye[1] - max_dist:
                    nearby_eyes.append(e)
                else:
                    break
            nearby_eyes_per_eye.append(nearby_eyes)
        nearby_eyes_per_eye = list(filter(lambda nearby_eyes: len(nearby_eyes) >= 2, nearby_eyes_per_eye))

        # For each nearby eyes, keep the two eyes with most similar area.
        for i, nearby_eyes in enumerate(nearby_eyes_per_eye):
            # Sort by area
            nearby_eyes = sorted(nearby_eyes, key=lambda eye: eye[2] * eye[3])
            best_index = 0
            smallest_delta = 1e9
            for i in range(len(nearby_eyes) - 1):
                eye1 = nearby_eyes[i]
                eye2 = nearby_eyes[i + 1]

                delta = abs(eye1[2] * eye1[3] - eye2[2] * eye2[3])
                if delta < smallest_delta:
                    best_index = i
                    smallest_delta = delta
            nearby_eyes_per_eye[i] = nearby_eyes[i], nearby_eyes[i + 1]

        # Take the eyes closest to 2/3 of the way up the face
        two_thirds_face_y = face[1] + 2 / 3 * face[3]
        smallest_delta = 1e9
        best_eyes = []
        for nearby_eyes in nearby_eyes_per_eye:
            delta = min(map(lambda eye: abs(two_thirds_face_y - eye[1]), nearby_eyes))
            if delta < smallest_delta:
                smallest_delta = delta
                best_eyes = nearby_eyes
        return best_eyes


    @staticmethod
    def extract_eyes(grey_image):
        '''
        Exracts eyes from the given image.
        :param grey_image: A grey image
        :return: Returns a tuple of bounding boxes (x,y,w,h) for the eyes. Bounding
            boxes are relative to the provided image. Returns an empty list if no
            eyes found.
        '''
        faces = EyeExtractor.face_cascade.detectMultiScale(grey_image, 1.3, 5)
        face_eyes_pairs = []
        all_eyes = []
        for face_bb in faces:
            face_roi = EyeExtractor.__extract_roi(grey_image, face_bb)
            eyes = EyeExtractor.eye_cascade.detectMultiScale(face_roi)
            eyes = list(map(lambda eye: [face_bb[0] + eye[0], face_bb[1] + eye[1], eye[2], eye[3]], eyes))
            face_eyes_pairs.append((face_bb, eyes))
            all_eyes.extend(eyes)

        # Sort by face area. Eventually, we only take the biggest face.
        face_eyes_pairs.sort(key=lambda face_eyes: face_eyes[0][2] * face_eyes[0][3])
        eye_pairs = map(EyeExtractor.__get_best_two_eyes, face_eyes_pairs)
        # Only take pairs with 2 or more eyes.
        eye_pairs = list(filter(lambda eyes: len(eyes) >= 2, eye_pairs))

        if len(eye_pairs) > 0:
            return eye_pairs[-1], all_eyes
        return [], []


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes, all_eyes = EyeExtractor.extract_eyes(grey)
        for ex,ey,ew,eh in all_eyes:
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0,0,255), 2)
        for ex,ey,ew,eh in eyes:
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
        cv2.imshow('img', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
