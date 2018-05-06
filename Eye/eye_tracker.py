import cv2
import math
import numpy as np


class EyeTracker:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    MAX_HIST = 5
    MAX_EYE_PCT_DIST = 0.25

    def __init__(self):
        self.smoothed_eyes = [[0, 0, 0, 0], [0, 0, 0, 0]]
        self.past_eyes = []
        self.smoothed_face = [0, 0, 0, 0]
        self.past_faces = []

    def update(self, grey_image):
        '''
        Updates the eye tracker with new observations from the grey image
        '''
        faces = EyeTracker.face_cascade.detectMultiScale(grey_image, 1.3, 5)
        eye_candidate_pairs = EyeTracker.__get_eye_candidate_pairs(grey_image, faces)

        self.__update_smoothed_eyes(eye_candidate_pairs)
        self.__update_smoothed_face(faces)


    def get_best_eyes(self):
        return self.smoothed_eyes

    def get_best_face(self):
        return self.smoothed_face

    def __update_smoothed_eyes(self, eye_candidate_pairs):
        best_delta = 1e9
        best_pair = None
        for eye_candidate_pair in eye_candidate_pairs:
            if eye_candidate_pair[0][0] < eye_candidate_pair[1][0]:
                left_eye, right_eye = eye_candidate_pair
            else:
                right_eye, left_eye = eye_candidate_pair
            left_delta = math.sqrt((left_eye[0] - self.smoothed_eyes[0][0]) ** 2 + (left_eye[1] - self.smoothed_eyes[0][1]) ** 2)
            right_delta = math.sqrt((right_eye[0] - self.smoothed_eyes[1][0]) ** 2 + (right_eye[1] - self.smoothed_eyes[1][1]) ** 2)
            delta = left_delta + right_delta
            if delta < best_delta:
                best_delta = delta
                best_pair = (left_eye, right_eye)

        if best_pair is not None:
            if len(self.past_eyes) == EyeTracker.MAX_HIST:
                popped_eyes = self.past_eyes.pop(0)
                for i in range(len(self.smoothed_eyes)):
                    for j in range(len(self.smoothed_eyes[i])):
                        self.smoothed_eyes[i][j] -= popped_eyes[i][j] / EyeTracker.MAX_HIST
            elif len(self.past_eyes) > 0:
                for i in range(len(self.smoothed_eyes)):
                    for j in range(len(self.smoothed_eyes[i])):
                        self.smoothed_eyes[i][j] *= len(self.past_eyes) / (len(self.past_eyes) + 1)

            self.past_eyes.append(best_pair)
            for i in range(len(self.smoothed_eyes)):
                for j in range(len(self.smoothed_eyes[i])):
                    self.smoothed_eyes[i][j] += best_pair[i][j] / len(self.past_eyes)


    def __update_smoothed_face(self, faces):
        '''
        Update the smoothed face based on the new faces and updated smoothed eyes.
        :param faces: A list of face bounding boxes.
        '''
        def contains_smoothed_eyes(face):
            '''
            Returns whether the face bounding box contains smoothed eyes.
            :param face: A face bounding box
            :return: Returns True if the face bounding box contains the smoothed eyes.
                Returns false otherwise.
            '''
            def contains_eye(face, eye):
                return face[0] <= eye[0] and \
                        face[1] <= eye[0] and \
                        face[0] + face[2] >= eye[0] + eye[2] and \
                        face[1] + face[3] >= eye[1] + eye[3]
            return contains_eye(face, self.smoothed_eyes[0]) and contains_eye(face, self.smoothed_eyes[1])


        filtered_faces = list(filter(contains_smoothed_eyes, faces))
        if len(filtered_faces) == 0:
            return
        sorted_faces = sorted(filtered_faces, key=lambda f: f[2] * f[3])
        face = sorted_faces[-1]

        if len(self.past_faces) == EyeTracker.MAX_HIST:
            popped_face = self.past_faces.pop(0)
            for i in range(len(self.smoothed_face)):
                self.smoothed_face[i] -= popped_face[i] / EyeTracker.MAX_HIST
        elif len(self.past_faces) > 0:
            for i in range(len(self.smoothed_face)):
                self.smoothed_face[i] *= len(self.past_faces) / (len(self.past_faces) + 1)

        self.past_faces.append(face)
        for i in range(len(self.smoothed_face)):
            self.smoothed_face[i] += face[i] / len(self.past_faces)


    @staticmethod
    def __extract_roi(img, roi_rect):
        x, y, w, h = roi_rect
        return img[y:y+h, x:x+w]


    @staticmethod
    def __get_eye_candidate_pairs_for_face(face_eyes):
        '''
        Returns the best two eyes for the given face and eyes.
        '''
        face, eyes = face_eyes

        # Discard eye candidates which do not have a pair at a similar y-coordinate.
        eyes.sort(key=lambda eye: eye[1])
        nearby_eyes_per_eye = []  # A list of nearby eyes for each eye
        for i, eye in enumerate(eyes):
            max_dist = eye[3] * EyeTracker.MAX_EYE_PCT_DIST
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
        eye_candidate_pairs = []
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
            eye_candidate_pairs.append((nearby_eyes[i], nearby_eyes[i + 1]))
        return eye_candidate_pairs


    @staticmethod
    def __get_eye_candidate_pairs(grey_image, faces):
        '''
        Exracts eyes from the given image.
        :param grey_image: A grey image
        :param faces: A list of face bounding boxes in the grey image.
        :return: Returns a tuple of bounding boxes (x,y,w,h) for the eyes. Bounding
            boxes are relative to the provided image. Returns an empty list if no
            eyes found.
        '''
        face_eyes_pairs = []
        all_eyes = []
        for face_bb in faces:
            face_roi = EyeTracker.__extract_roi(grey_image, face_bb)
            eyes = EyeTracker.eye_cascade.detectMultiScale(face_roi)
            eyes = list(map(lambda eye: [face_bb[0] + eye[0], face_bb[1] + eye[1], eye[2], eye[3]], eyes))
            face_eyes_pairs.append((face_bb, eyes))
            all_eyes.extend(eyes)

        # Sort by face area. Eventually, we only take the biggest face.
        face_eyes_pairs.sort(key=lambda face_eyes: face_eyes[0][2] * face_eyes[0][3])
        eye_pair_lists = map(EyeTracker.__get_eye_candidate_pairs_for_face, face_eyes_pairs)
        # Only take pairs with at least one pair
        eye_pair_lists = list(filter(lambda eye_pairs: len(eye_pairs) >= 1, eye_pair_lists))

        all_pairs = []
        for eye_pair_list in eye_pair_lists:
            all_pairs.extend(eye_pair_list)
        return all_pairs


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    eye_tracker = EyeTracker()
    while True:
        ret, frame = cap.read()
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eye_tracker.update(grey)
        eyes = eye_tracker.get_best_eyes()
        x,y,w,h = eye_tracker.get_best_face()
        x,y,w,h = int(x), int(y), int(w), int(h)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
        for ex,ey,ew,eh in eyes:
            ex, ey, ew, eh = int(ex), int(ey), int(ew), int(eh)
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
        cv2.imshow('img', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
