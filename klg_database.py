import json

from matplotlib.backend_tools import cursors
from pymongo import MongoClient
etl_connection = MongoClient("mongodb://localhost:27017/")
local_connection = MongoClient("mongodb://localhost:27017")

with open('src/data/debtors.json', 'r') as f:
    debtors = json.loads(f.read())

chains = {
    "bnb": {"$gte": 39117555},
    "ethereum": {"$gte": 19970000},
    "polygon": {"$gte": 57535325},
    "arbitrum": {"$gte": 215859437}
}

for idx in range(0, len(debtors), 1000):
    for key, value in chains.items():
        if key != "bnb":
            collection = f"{key}_blockchain_etl"
        else:
            collection = "blockchain_etl"
        etl_database = etl_connection[collection]
        cursor = list(etl_database["transactions"].find({
            "from_address": {"$in": debtors[idx:idx+1000]},
            "block_number": value
        }))
        to_tx = etl_database["transactions"].find({
            "to_address": {"$in": debtors[idx:idx + 1000]},
            "block_number": value
        })
        for item in to_tx:
            if item in cursor: continue
            cursor.append(item)
        with open(f'src/data/transactions/{key}/{idx}.json', 'w') as f:
            json.dump(cursor,f, indent=1)
