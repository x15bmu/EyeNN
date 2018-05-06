import cv2
from GazeCapture.pytorch.ITrackerModel import ITrackerModel
from Eye.eye_tracker import EyeTracker
import time
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as sio
import torchvision.transforms as transforms


def loadMetadata(filename, silent = False):
    try:
        # http://stackoverflow.com/questions/6273634/access-array-contents-from-a-mat-file-loaded-using-scipy-io-loadmat-python
        if not silent:
            print('\tReading metadata from %s...' % filename)
        metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
    except:
        print('\tFailed to read the meta file "%s"!' % filename)
        return None
    return metadata


class SubtractMean(object):
    """Normalize an tensor image with mean.
    """

    def __init__(self, meanImg):
        self.meanImg = transforms.ToTensor()(meanImg)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        return tensor.sub(self.meanImg)


class ITrackerData:
    MEAN_PATH = './GazeCapture/models/mean_images/'

    def __init__(self, imSize=(224,224), gridSize=(25, 25)):

        self.imSize = imSize
        self.gridSize = gridSize

        self.faceMean = loadMetadata(os.path.join(ITrackerData.MEAN_PATH, 'mean_face_224.mat'))['image_mean']
        self.eyeLeftMean = loadMetadata(os.path.join(ITrackerData.MEAN_PATH, 'mean_left_224.mat'))['image_mean']
        self.eyeRightMean = loadMetadata(os.path.join(ITrackerData.MEAN_PATH, 'mean_right_224.mat'))['image_mean']

        self.transformFace = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Scale(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.faceMean),
        ])
        self.transformEyeL = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Scale(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.eyeLeftMean),
        ])
        self.transformEyeR = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Scale(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.eyeRightMean),
        ])

    def process_face(self, face):
        return self.transformFace(face)

    def process_left_eye(self, left_eye):
        return self.transformEyeL(left_eye)

    def process_right_eye(self, right_eye):
        return self.transformEyeR(right_eye)


class Main:
    CHECKPOINTS_PATH = './GazeCapture/pytorch'
    IMG_SIZE = (224, 224)

    def __init__(self):
        self.eye_tracker = EyeTracker()
        self.model = ITrackerModel()
        self.model.double()
        self.tracker_data = ITrackerData()
        self.avg_time = 0
        self.num_times = 0

        self.__load_model()

        self.xs = []
        self.ys = []

    def process_frame(self, frame):
        start = time.time()
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.eye_tracker.update(grey_frame)

        face_bb = self.eye_tracker.get_best_face()
        right_eye_bb, left_eye_bb = self.eye_tracker.get_best_eyes()

        face = Main.__extract_color_roi(frame, face_bb)
        left_eye = Main.__extract_color_roi(frame, left_eye_bb)
        right_eye = Main.__extract_color_roi(frame, right_eye_bb)

        if face_bb[2] == 0 or face_bb[3] == 0:
            return

        face_grid = np.asarray(Main.__compute_face_grid(face_bb, frame))
        face_grid = torch.from_numpy(face_grid).double()

        # cv2.imshow('face', cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        # cv2.imshow('left_eye', cv2.cvtColor(left_eye, cv2.COLOR_RGB2BGR))
        # cv2.imshow('right_eye', cv2.cvtColor(right_eye, cv2.COLOR_RGB2BGR))

        face = self.tracker_data.process_face(face).double()
        left_eye = self.tracker_data.process_left_eye(left_eye).double()
        right_eye = self.tracker_data.process_right_eye(right_eye).double()

        face = face.unsqueeze(0)
        left_eye = left_eye.unsqueeze(0)
        right_eye = right_eye.unsqueeze(0)
        face_grid = face_grid.unsqueeze(0)

        face = Variable(face)
        left_eye = Variable(left_eye)
        right_eye = Variable(right_eye)
        face_grid = Variable(face_grid)

        out_start = time.time()
        output = self.model(face, left_eye, right_eye, face_grid)
        end = time.time()
        print(output[0])
        if self.avg_time == 0:
            self.avg_time = end - start
        else:
            self.avg_time *= self.num_times / (self.num_times + 1)
            self.avg_time += (end - start) / (self.num_times + 1)
        self.num_times += 1

        self.xs.append(-output.data[0][0])
        self.ys.append(output.data[0][1])
        plt.plot(self.xs, self.ys, '-ob')
        plt.draw()
        plt.show(block=False)

        color_corrected_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        eyes = (left_eye_bb, right_eye_bb)
        x,y,w,h = face_bb
        x,y,w,h = int(x), int(y), int(w), int(h)
        cv2.rectangle(color_corrected_frame, (x, y), (x+w, y+h), (255,0,0), 2)

        ex,ey,ew,eh = eyes[0]
        ex, ey, ew, eh = int(ex), int(ey), int(ew), int(eh)
        cv2.rectangle(color_corrected_frame, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

        ex,ey,ew,eh = eyes[1]
        ex, ey, ew, eh = int(ex), int(ey), int(ew), int(eh)
        cv2.rectangle(color_corrected_frame, (ex, ey), (ex+ew, ey+eh), (0,0,255), 2)

        cv2.imshow('frame', color_corrected_frame)

    def __load_model(self, filename='checkpoint.pth.tar'):
        filename = os.path.join(Main.CHECKPOINTS_PATH, filename)
        if not os.path.isfile(filename):
            print('Warning: Could not read checkpoint!');
            return

        saved = torch.load(filename, map_location='cpu')
        print('Loading checkpoint for epoch %05d with error %.5f...' % (saved['epoch'], saved['best_prec1']))
        state = saved['state_dict']
        try:
            self.model.module.load_state_dict(state)
        except:
            self.model.load_state_dict(state)

    @staticmethod
    def __extract_color_roi(frame, roi):
        x,y,w,h = [int(v) for v in roi]
        return frame[y:y+h, x:x+w]

    @staticmethod
    def __compute_face_grid(face_bb, frame):
        """
        Computes the face grid  as defined by the NN definition.
        The face grid is a 25x25 grid which specifies
        where the face is located in the grid.
        """
        h, w = frame.shape[:2]
        gx = int(face_bb[0] / w * 25)
        gy = int(face_bb[1] / h * 25)
        gw = int(face_bb[2] / w * 25)
        gh = int(face_bb[3] / h * 25)

        grid = np.zeros((25, 25))
        for x in range(len(grid)):
            for y in range(len(grid[x])):
                if x > gx and x <= gx + gw and y > gy and y <= gy + gh:
                    grid[x][y] = 1
        return grid


if __name__ == '__main__':
    main = Main()
    #cap = cv2.VideoCapture('./Eye/test.mov')
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()

        main.process_frame(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

