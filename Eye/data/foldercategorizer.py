from GazeCapture.pytorch.ITrackerData import ITrackerData
from typing import Tuple


def get_dirs_with_type(split_type: str) -> Tuple[str]:
    """
    Get a list of directories with the given type.
    :param split_type: The type of split: train, val, or test.
    :return: A list of directories.
    """
    if split_type not in ('train', 'val', 'test'):
        raise ValueError('Expected: train, val, or test')

    data = ITrackerData(split_type)

    dirs = set()

    for i in range(len(data.indices)):
        index = data.indices[i]
        label_rec_num = data.metadata['labelRecNum'][index]
        dirs.add(label_rec_num)

    return tuple(sorted(dirs))


if __name__ == '__main__':
    tars = ['%05d.tar.gz' % d for d in get_dirs_with_type('val')]
    print(' '.join(tars[:20]))
