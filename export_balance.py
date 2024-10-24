import json
import os
import time

from dotenv import load_dotenv

from src.backtest.database.klg_mongodb import MongoDbKLG
from src.backtest.database.mongodb import MongoDB
from src.backtest.export_balance_wallets import ExportBalanceWalletJob

from src.constants.constants_v3 import Chain
from src.utils.logger_utils import get_logger

load_dotenv()

with open("src/backtest/balance_timestamp.json", "r") as f:
    wallets = json.loads(f.read())
for chain in ["0x38", "0x1", "0x89", "0xa4b1"]:
    wallet_timestamps = {key: value[chain] for key, value in wallets.items()}
    prefix = Chain.prefix.get(chain)
    _importer = MongoDB(os.environ.get("MONGO_MAIN"), "blockchain_etl", prefix)
    _exporter = MongoDB(os.environ.get("LOCAL_DB"), "knowledge_graph")
    _klg_db = MongoDbKLG(os.environ.get("ENTITIES_DB"), chain_id=chain)
    _protocol_db = MongoDB(os.environ.get("DAPP_INFO_DB"), "SmartContractLabel")
    if not prefix:
        prefix = "bsc"
    provider = os.environ.get(f"{prefix.upper()}_PROVIDER")
    print(f"Start run chain {chain}")
    begin=time.time()
    job = ExportBalanceWalletJob(
        chain_id=chain,
        importer=_importer,
        exporter=_exporter,
        klg_db=_klg_db,
        protocol_db=_protocol_db,
        batch_size=1000,
        wallets=wallet_timestamps,
        provider=provider
    )
    job.run()
    print(f"Run in {time.time()-begin}")