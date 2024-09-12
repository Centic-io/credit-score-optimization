from collections import defaultdict
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def column_chart(data):
    keys = [str(key) for key in data.keys()]
    values = [value for value in data.values()]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(keys, values, color="skyblue")
    plt.xlabel("Accuracy")
    plt.ylabel("Percentage")
    plt.xticks(rotation=45, ha="right")

    # Adding the value on top of each column
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()
    
def column_chart2(data, label):
    # Create a temporary list containing values from 0 to 4
    temp_keys = [str(i) for i in range(5)]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(temp_keys, [data.get(int(key), 0) for key in temp_keys], color='skyblue') # Use get() to retrieve value for each key or 0 if key not present
    plt.xlabel(f'{label} Pairs')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45, ha='right')

    # Adding the value on top of each column
    for key, value in data.items():
        index = temp_keys.index(str(key))
        plt.text(index, value, f'{value:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
    
def heat_map(input_dict):
    df = pd.DataFrame(input_dict).fillna(0)
    df_sorted = df.sort_index(ascending=False)
    # Create heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_sorted, annot=True, fmt="g")
    # Add labels and title
    plt.xlabel("Pair Label")
    plt.ylabel("Label")
    plt.title("Heatmap from Nested Dictionary")
    plt.show()

def reporting(first_label, second_label, pred_label):
    count = {}  #  key: pair label, value: number of accuracy pred_label
    count2 = defaultdict(int)  # key: pair label, value: total number of pair_label
    count3 = defaultdict(
        lambda: defaultdict(int)
    )  # key: pair_label, value: {key: unique label, value: number of label}

    # Xác định các cặp label

    for i in range(len(first_label)):
        pair = (
            min(first_label[i], second_label[i]),
            max(first_label[i], second_label[i]),
        )
        if count.get(pair) is None: 
            count[pair] = 0
        count2[pair] += 1

        count3[pair][pred_label[i]] += 1

        if pred_label[i] == pair[0] or pred_label[i] == pair[1]:

            count[pair] += 1

    sorted_dict = dict(sorted(count3.items(), key=lambda x: x[0]))
    heat_map(sorted_dict)
    # Report 1: Tỉ lệ đúng của từng cặp nhãn
    result_1 = defaultdict(lambda: defaultdict(float))
    for key in count.keys():
        result_1[key] = round(count[key] / count2[key], 2)
        # print(key, count[key] / count2[key])
    column_chart(result_1)
    # Report 2: Tỉ lệ phân bổ các nhãn của từng cặp nhãn
    result_2 = defaultdict(lambda: defaultdict(float))
    for key, value in sorted_dict.items():
        for key2, value2 in value.items():
            result_2[key][key2] = round(value2 / count2[key] * 100, 2)
    for key, value in result_2.items():
        column_chart2(value, key)
