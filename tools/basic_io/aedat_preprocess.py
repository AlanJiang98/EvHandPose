import numpy as np
from dv import AedatFile
from dv import LegacyAedatFile

def extract_data_from_aedat2(path: str):
    '''
    extract events from aedat2 data
    :param path:
    :return:
    '''
    with LegacyAedatFile(path) as f:
        events = []
        for event in f:
            events.append(np.array([[event.x, event.y, event.polarity, event.timestamp]], dtype=np.int64))
        if not events:
            return None
        else:
            events = np.vstack(events)
            return events
    return FileNotFoundError('Path {} is unavailable'.format(path))


def extract_data_from_aedat4(path: str, is_event: bool = True, is_aps: bool = True, is_trigger: bool = True):
    '''
    :param path: path to aedat4 file
    :return: events numpy array, aps numpy array
        event:
        # Access information of all events by type
        timestamps, x, y, polarities = events['timestamp'], events['x'], events['y'], events['polarity']
        is_aps: list of frames
        is_trigger: list of triggers
        # EXTERNAL_INPUT_RISING_EDGE->2, EXTERNAL_INPUT1_RISING_EDGE->6, EXTERNAL_INPUT2_RISING_EDGE->9
        # EXTERNAL_INPUT1_PULSE->8, TIMESTAMP_RESET->1, TIMESTAMP_WRAP->0, EXTERNAL_INPUT1_FALLING_EDGE->7
    '''
    with AedatFile(path) as f:
        events, frames, triggers = None, [], [[], [], [], [], [], [], []]
        id2index = {2: 0, 6: 1, 9: 2, 8: 3, 1: 4, 0: 5, 7: 6}
        if is_event:
            events = np.hstack([packet for packet in f['events'].numpy()])
        if is_aps:
            for frame in f['frames']:
                frames.append(frame)
        if is_trigger:
            for i in f['triggers']:
                if i.type in id2index.keys():
                    triggers[id2index[i.type]].append(i)
                else:
                    print("{} at {} us is the new trigger type in this aedat4 file".format(i.type, i.timestamp))
        return events, frames, triggers
    return FileNotFoundError('Path {} is unavailable'.format(path))


