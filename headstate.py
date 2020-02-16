eye_close_thres = 0.02
mouth_close_thres = 0.4


def mouth_judge(p1, p2, p3, p4):
    return ((p3 - p2) / (p4 - p1)) < mouth_close_thres