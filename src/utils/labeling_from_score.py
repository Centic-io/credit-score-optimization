import numpy as np
def labeling(scores):
    scores =  np.round(scores).astype(int)
    label = []
    for score in scores:
        if score < 580:
            label.append(0)
        elif score >= 580 and score < 670:
            label.append(1)
        elif score >= 670 and score < 740:
            label.append(2)
        elif score >= 740 and score < 800:
            label.append(3)
        elif score >= 800:
            label.append(4)
    return np.array(label)