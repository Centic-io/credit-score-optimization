import numpy as np
from collections import defaultdict

def new_accuracy(first_label, second_label, y_pred):
    condition = np.logical_or(y_pred == first_label, y_pred == second_label)
    count = np.sum(condition)
    accuracy = count / len(first_label)
    return accuracy


# Custom F-score
def precision_recall(first_label, second_label, predicted_labels):
    count3 = defaultdict(
        lambda: defaultdict(int)
    )  # key: pair_label, value: {key: unique label, value: number of label}

    # Xác định các cặp label

    for i in range(len(first_label)):
        pair = (
            min(first_label[i], second_label[i]),
            max(first_label[i], second_label[i]),
        )
        count3[pair][predicted_labels[i]] += 1

    count_TP = defaultdict(int)
    count_FP = defaultdict(int)
    count_FN = defaultdict(int)
    count_total = defaultdict(int)

    for pair, value in count3.items():
        for key, count in value.items():
            if key in pair:
                count_TP[key] += count
            else:
                count_FP[key] += count
                count_FN[pair[0]] += count
                if pair[0] != pair[1]:
                    count_FN[pair[1]] += count
            count_total[key] += count
    precision = {}
    recall = {}
    f1 = {}
    for label in count_TP.keys():
        tp = count_TP[label]
        fp = count_FP[label]
        fn = count_FN[label]

        if tp + fp > 0:
            precision[label] = tp / (tp + fp)
        else:
            precision[label] = 0

        if tp + fn > 0:
            recall[label] = tp / (tp + fn)
        else:
            recall[label] = 0

        if precision[label] + recall[label] > 0:
            f1[label] = (
                2
                * (precision[label] * recall[label])
                / (precision[label] + recall[label])
            )
        else:
            f1[label] = 0
    avg_precision = sum(precision.values()) / len(precision) if precision else 0
    avg_recall = sum(recall.values()) / len(recall) if recall else 0
    avg_f1 = sum(f1.values()) / len(f1) if f1 else 0

    total_instances = sum(count_total.values())
    weighted_precision = (
        sum(precision[label] * count_total[label] for label in precision.keys())
        / total_instances
        if total_instances > 0
        else 0
    )
    weighted_recall = (
        sum(recall[label] * count_total[label] for label in recall.keys())
        / total_instances
        if total_instances > 0
        else 0
    )
    weighted_f1 = (
        sum(f1[label] * count_total[label] for label in f1.keys()) / total_instances
        if total_instances > 0
        else 0
    )

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Average Precision:", avg_precision)
    print("Average Recall:", avg_recall)
    print("Average F1 Score:", avg_f1)
    print("Weighted Precision:", weighted_precision)
    print("Weighted Recall:", weighted_recall)
    print("Weighted F1 Score:", weighted_f1)
