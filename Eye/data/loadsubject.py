import json
import numpy as np
import os
import os.path

def __get_detections_dict(inp):
    ret = {}
    ret['x'] = np.array([float(x) for x in inp['X']])
    ret['y'] = np.array([float(y) for y in inp['Y']])
    ret['w'] = np.array([float(w) for w in inp['W']])
    ret['h'] = np.array([float(h) for h in inp['H']])

    valids = np.array([bool(b) for b in inp['IsValid']])
    ret['x'][~valids] = float('nan')
    ret['y'][~valids] = float('nan')
    ret['w'][~valids] = float('nan')
    ret['h'][~valids] = float('nan')
    return ret


def load_subject(path):
    output = {}
    with open(os.path.join(path, 'appleFace.json')) as f:
        inp = json.load(f)
        output['appleFace'] = __get_detections_dict(inp)
    with open(os.path.join(path, 'appleLeftEye.json')) as f:
        inp = json.load(f)
        output['appleLeftEye'] = __get_detections_dict(inp)
    with open(os.path.join(path, 'appleRightEye.json')) as f:
        inp = json.load(f)
        output['appleRightEye'] = __get_detections_dict(inp)
    with open(os.path.join(path, 'faceGrid.json')) as f:
        inp = json.load(f)
        output['faceGrid'] = __get_detections_dict(inp)
    with open(os.path.join(path, 'frames.json')) as f:
        inp = json.load(f)
        output['frames'] = inp
    # TODO Determine if we need to load the other things. Maybe not.
    return output

