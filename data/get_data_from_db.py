import json
import pymongo
from config import MongoDBConfig
import pandas as pd
import numpy as np


def get_30_value_from_dict(dict_data):
    if dict_data is None or len(dict_data) == 0:
        return None
    sorted_data = sorted(dict_data.items(), key=lambda x: x[0], reverse=True)
    top_30_keys = [item[0] for item in sorted_data[: min(30, len(sorted_data))]]
    top_30_values = [dict_data[key] for key in top_30_keys]
    array_values = np.array(top_30_values)
    return np.mean(array_values)


# config
config = MongoDBConfig()

# mongoDB

list_hosts = config.HOST.split(",")
hosts_port = ""
for host in list_hosts:
    hosts_port += f"{host}:{config.PORT},"
hosts_port = hosts_port.rstrip(",")
mongo_uri = f"mongodb://{config.USERNAME}:{config.PASSWORD}@{hosts_port}/"

# connect
client = pymongo.MongoClient(mongo_uri)
db = client["knowledge_graph"]
collection = db["wallets"]

# depositors
df = pd.read_csv("all_data.csv")
depositers = df["address"]
depositers = ["0x1_" + depositer for depositer in depositers]


projection = {
    "address": 1,
    "balanceInUSD": 1,
    "borrowInUSD": 1,
    "depositInUSD": 1,
    "createdAt": 1,
    "numberOfLiquidation": 1,
    "totalValueOfLiquidation": 1,
    "frequencyOfDappTransactions": 1,
    "numberOfInteractedDapps": 1,
    "numberOfReputableDapps": 1,
    "typesOfInteractedDapps": 1,
    "borrowChangeLogs": 1,
    "depositChangeLogs": 1,
    "balanceChangeLogs": 1,
    "dailyNumberOfTransactions": 1,
    "dailyTransactionAmounts": 1,
    "numberCallsTopContracts": 1,
}

filter = {"_id": {"$in": depositers}}
cursor = collection.find(filter, projection)

list_data = []
cursor = list(cursor)
for i in range(len(cursor)):
    ##
    cursor[i]["frequencyOfTransaction"] = (
        get_30_value_from_dict(cursor[i]["dailyNumberOfTransactions"])
        if cursor[i].get("dailyNumberOfTransactions") is not None
        else None
    )
    #
    # cursor[i]["frequencyCallsTopContracts"] = (
    #     get_30_value_from_dict(cursor[i]["numberCallsTopContracts"])
    #     if cursor[i].get("numberCallsTopContracts") is not None
    #     else 0
    # )
    #
    cursor[i]["totalAsset"] = (
        cursor[i]["balanceInUSD"] + cursor[i]["depositInUSD"] - cursor[i]["borrowInUSD"]
        if cursor[i].get("balanceInUSD") is not None
        and cursor[i].get("depositInUSD") is not None
        and cursor[i].get("borrowInUSD") is not None
        else None
    )
    #
    cursor[i]["frequencyMountOfTransaction"] = get_30_value_from_dict(
        cursor[i]["dailyTransactionAmounts"]
        if cursor[i].get("dailyTransactionAmounts") is not None
        else None
    )
    ##
    cursor[i]["averageBalance"] = (
        get_30_value_from_dict(cursor[i]["balanceChangeLogs"])
        if cursor[i].get("balanceChangeLogs") is not None
        else None
    )
    # avg_borrow = get_30_value_from_dict(
    #     cursor[i]["borrowChangeLogs"] if cursor[i].get("borrowChangeLogs") is not None else None
    # )
    # avg_deposit = get_30_value_from_dict(
    #     cursor[i]["depositChangeLogs"] if cursor[i].get("depositChangeLogs") is not None else None
    # )


new_df = pd.DataFrame(cursor)

delete_column = [
    "_id",
    "balanceChangeLogs",
    "depositChangeLogs",
    "borrowChangeLogs",
    "dailyNumberOfTransactions",
    "dailyTransactionAmounts",
    "numberCallsTopContracts"
]
for col in delete_column:
    new_df = new_df.drop(col, axis=1)

# Merge & write to csv file
csv_file_path = "Lending-Data-Ethereum.csv"
new_df.to_csv(csv_file_path, index=False)
# with open("tmp.json", "w") as f:
#     json.dump(cursor, f, indent=4)
