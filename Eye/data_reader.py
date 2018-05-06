from GazeCapture.pytorch.ITrackerData import ITrackerData
from GazeCapture.pytorch.ITrackerModel import ITrackerModel
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import os.path
import random
import shutil
import torch
from torch.autograd import Variable



DATASET_PATH = os.path.join(os.getcwd(), 'Eye/data/processed')
CHECKPOINTS_PATH = './GazeCapture/pytorch'
SUPPORTED_INDICES = set([135, 213, 267, 507, 547])
DO_GRAPH = False
model = ITrackerModel()
# xs = []
# ys = []
lines = []
colors = []
fig = None
ax = None

image_errors = dict()

def process_data(data):
    '''
    Processes the image and returns the l2 error for that image.
    '''
    global model, xs, ys, fig, ax
    row, face, left_eye, right_eye, face_grid, gaze = data

    # Code for shifting face rect. Used to determine how the accuracy of the face rect affects the accuracy
    # of the final prediction
    # face_grid_rect = face_grid.numpy().reshape(25, 25)
    # direction = random.randint(0, 3)
    # if direction == 0:
    #     face_grid_rect = np.concatenate((face_grid_rect[-1, np.newaxis, :], face_grid_rect[:-1, :]))
    # elif direction == 1:
    #     face_grid_rect = np.concatenate((face_grid_rect[:, -1, np.newaxis], face_grid_rect[:, :-1]), axis=1)
    # elif direction == 2:
    #     face_grid_rect = np.concatenate((face_grid_rect[1:, :], face_grid_rect[0, np.newaxis, :]))
    # else:
    #     face_grid_rect = np.concatenate((face_grid_rect[:, 1:], face_grid_rect[:, 0, np.newaxis]), axis=1)

    # face_grid = torch.from_numpy(face_grid_rect.reshape(625))
    face = face.unsqueeze(0)
    left_eye = left_eye.unsqueeze(0)
    right_eye = right_eye.unsqueeze(0)
    face_grid = face_grid.unsqueeze(0)

    face = Variable(face)
    left_eye = Variable(left_eye)
    right_eye = Variable(right_eye)
    face_grid = Variable(face_grid)
    output = model(face, left_eye, right_eye, face_grid)

    #print('Output', output.data[0])
    #print('Gaze', gaze)
    error = output.data[0] - gaze
    error_norm = error.norm()
    print('Error', error.norm())

    if DO_GRAPH:
        lines.append((gaze.tolist(), output.data[0].tolist()))
        if error_norm < 1:
            colors.append((0, 1, 0))
        elif error_norm < 2:
            colors.append((1, 1, 0))
        elif error_norm < 3:
            colors.append((1, 0.65, 0))
        else:
            colors.append((1, 0, 0))

        lc = LineCollection(lines, colors=colors)
        if fig is None:
            fig, ax = plt.subplots()
        ax.add_collection(lc)
        ax.autoscale()
        plt.draw()
        plt.show(block=False)
        plt.pause(0.001)
    return error_norm


def load_model(filename='checkpoint.pth.tar'):
    global model, CHECKPOINTS_PATH

    filename = os.path.join(CHECKPOINTS_PATH, filename)
    if not os.path.isfile(filename):
        print('Warning: Could not read checkpoint!');
        return

    saved = torch.load(filename, map_location='cpu')
    print('Loading checkpoint for epoch %05d with error %.5f...' % (saved['epoch'], saved['best_prec1']))
    state = saved['state_dict']
    try:
        model.module.load_state_dict(state)
    except:
        model.load_state_dict(state)

data = ITrackerData(split='val', imSize = (224, 224))
index = 0
load_model()

# Eye conv layers are 3, 7, 11, 13
for idx, m in enumerate(model.modules()):
    print(idx, '->', m)
raise Exception("hi")

found_indices = set()
errors = []
avg_error = 0
e_count = 0
record_nums = set()
for index in range(len(data.indices)):
    data_index = data.indices[index]
    label_rec_num = data.metadata['labelRecNum'][data_index]
    record_nums.add(label_rec_num)
    if label_rec_num in SUPPORTED_INDICES:
        print("Data Index: ", data_index)
        print("Rec num: ", label_rec_num)
        found_indices.add(label_rec_num)
        error = process_data(data[index])
        errors.append(error)
        e_count += 1
        avg_error = avg_error * (e_count - 1) / e_count + error / e_count

        rect_path = os.path.join(DATASET_PATH, '%05d/rect/%05d.jpg' % (data.metadata['labelRecNum'][index], data.metadata['frameIndex'][index]))
        error_images_dir = os.path.join(DATASET_PATH, 'error_images')
        os.makedirs(error_images_dir, exist_ok=True)
        error_image_path = os.path.join(error_images_dir, '%5f.jpg' % error)
        # print("Writing image: ", error_image_path)
        # shutil.copy2(rect_path, error_image_path)

        print("Average error: ", avg_error)

plt.hist(errors)
plt.show()

#print('Missing indices: ', SUPPORTED_INDICES - found_indices)

