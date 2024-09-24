import numpy as np
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras.metrics import Metric

def new_accuracy(first_label, second_label, y_predict):
    condition = np.logical_or(y_predict == first_label, y_predict == second_label)
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

    count_tp = defaultdict(int)
    count_fp = defaultdict(int)
    count_fn = defaultdict(int)
    count_total = defaultdict(int)

    for pair, value in count3.items():
        for key, count in value.items():
            if key in pair:
                count_tp[key] += count
            else:
                count_fp[key] += count
                count_fn[pair[0]] += count
                if pair[0] != pair[1]:
                    count_fn[pair[1]] += count
            count_total[key] += count
    precision = {}
    recall = {}
    f1 = {}
    for label in count_tp.keys():
        tp = count_tp[label]
        fp = count_fp[label]
        fn = count_fn[label]

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

def normal(arr):
    sum_ = np.sum(arr)
    arr = arr / sum_
    return arr

class CustomSparseCategoricalCrossEntropy(tf.keras.losses.Loss):
    def __init__(
        self,
        from_logits=False,
        reduction=tf.keras.losses.Reduction.AUTO,
        name="custom_sparse_categorical_crossentropy",
    ):
        super().__init__(reduction=reduction, name=name)
        self.from_logits = from_logits

    def call(self, y_true, y_predict):
        # Calculate the standard sparse categorical cross entropy loss
        min_losses = []
        first_label = tf.cast(y_true[:, 0], tf.int32)
        second_label = tf.cast(y_true[:, 1], tf.int32)

        first_scce_loss = tf.keras.losses.sparse_categorical_crossentropy(
            first_label, y_predict, from_logits=self.from_logits
        )
        second_scce_loss = tf.keras.losses.sparse_categorical_crossentropy(
            second_label, y_predict, from_logits=self.from_logits
        )
        return tf.reduce_min(tf.stack([first_scce_loss, second_scce_loss], axis=0), axis=0)

class CustomAccuracy(Metric):
    def __init__(self, **kwargs):
        super(CustomAccuracy, self).__init__(**kwargs)
        self.correct = self.add_weight("correct", initializer="zeros")
        self.total = self.add_weight("total", initializer="zeros")

    def update_state(self, y_true, y_predict, sample_weight=None):
        y_predict = tf.argmax(y_predict, axis=1)
        # Cast y_true to the same dtype as y_predict
        first_label = tf.cast(y_true[:, 0], y_predict.dtype)
        second_label = tf.cast(y_true[:, 1], y_predict.dtype)
        # Check the equality of the prediction and truth
        values = tf.cast(tf.logical_or(tf.equal(y_predict, first_label), tf.equal(y_predict, second_label)), 'float32')
        self.correct.assign_add(tf.reduce_sum(values))
        self.total.assign_add(tf.cast(tf.size(first_label), 'float32'))

    def result(self):
        return self.correct / self.total

    def reset_state(self):
        self.correct.assign(0)
        self.total.assign(0)