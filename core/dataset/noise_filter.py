import numpy as np

def background_activity_filter(events: np.ndarray, size=(260, 346), delta_t=2000, N_thre=4):
    tmp_events = events.copy().astype(np.int32)
    legal_x = (tmp_events[:, 0] >= 0) * (tmp_events[:, 0] < size[1])
    legal_y = (tmp_events[:, 1] >= 0) * (tmp_events[:, 1] < size[0])
    legal_xy = (legal_y*legal_x).sum()
    print(legal_xy)
    assert legal_xy == events.shape[0]
    time_image = -delta_t * np.ones((size[0]+2, size[1]+2)) + events[0, 3] - 1
    is_noise = np.zeros(tmp_events.shape[0])
    index_x = np.array([-1, 0, 1, -1, 1, -1, 0, 1]) + 1
    index_y = np.array([-1, -1, -1, 0, 0, 1, 1, 1]) + 1
    for i in range(tmp_events.shape[0]):
        tmp = tmp_events[i, 3] - time_image[index_y + tmp_events[i, 1], index_x + tmp_events[i, 0]]
        tmp = (tmp < delta_t).sum()
        if tmp < N_thre:
            is_noise[i] = 1
        time_image[tmp_events[i, 1]+1, tmp_events[i, 0] + 1] = tmp_events[i, 3]
    bool_valid = (1-is_noise) == 1
    return bool_valid

