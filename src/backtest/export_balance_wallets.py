import json
import os
import time

from defi_services.constants.query_constant import Query
from defi_services.jobs.processors.multi_call_state_processor import MultiCallStateProcessor
from dotenv import load_dotenv

from src.backtest.cli_job import CLIJob
from src.backtest.database.klg_mongodb import MongoDbKLG
from src.backtest.database.memory_storage import MemoryStorage
from src.backtest.database.mongodb import MongoDB
from src.constants.constants_v3 import Chain
from src.constants.network_constants import NATIVE_TOKEN
from src.utils.logger_utils import get_logger

load_dotenv()
logger = get_logger("Balance Wallet")
class ExportBalanceWalletJob(CLIJob):
    def __init__(
            self,
            importer: MongoDB,
            exporter: MongoDB,
            klg_db: MongoDbKLG,
            protocol_db: MongoDB,
            wallets,
            batch_size,
            chain_id,
            provider
    ):
        super().__init__(retry=False)
        self.protocol_name = {}
        self.wallets = wallets
        self.protocol_db = protocol_db
        self.klg_db = klg_db
        self.batch_size = batch_size
        self.exporter = exporter
        self.importer = importer
        self.chain_id = chain_id
        self.local_storage = MemoryStorage.get_instance()
        self.multicall = MultiCallStateProcessor(
            provider_uri=provider,
            chain_id=chain_id
        )

    def _prepare_queries(self, wallets):
        cursor = self.exporter.get_documents("multichain_wallets",{"_id": {"$in": wallets}})
        self.queries = {}
        for wallet in cursor:
            if wallet.get("exported_chain", {}).get(self.chain_id):
                continue
            dict_tokens = wallet.get("tokens")
            if not dict_tokens:
                dict_tokens = wallet.get("depositTokens")
            if f"{self.chain_id}_{NATIVE_TOKEN}" not in dict_tokens:
                dict_tokens[f"{self.chain_id}_{NATIVE_TOKEN}"] = 1
            for key, value in dict_tokens.items():
                if not value:
                    continue
                chain_id, token = key.split('_')[0], key.split('_')[1]
                if chain_id != self.chain_id:
                    continue
                if token == self.chain_id:
                    continue
                for timestamp, block_number in self.wallets.get(wallet.get('_id')).items():
                    self.queries.update({
                        f"{timestamp}_{token}_{wallet.get('_id')}": {
                            "query_id": f"{timestamp}_{token}_{wallet.get('_id')}",
                            "entity_id": token,
                            "query_type": Query.token_balance,
                            "wallet": wallet.get('_id'),
                            "block_number": block_number
                        }
                    })

    def _execute(self, *args, **kwargs):
        keys = list(self.wallets.keys())
        for idx in range(0, len(keys), 100):
            begin = time.time()
            self._prepare_queries(keys[idx:idx+100])
            response_data = self.multicall.run(self.queries, batch_size=2000, max_workers=1, ignore_error=True)
            data = {}
            for key, value in response_data.items():
                split_key = key.split("_")
                address, token, timestamp = split_key[-1], split_key[1], split_key[0]
                if address not in data:
                    data[address] = {
                        "_id": address,
                        "tokenChangeLogs": {},
                        "exported_chain": {
                            self.chain_id: True
                        }
                    }
                token_key = f"{self.chain_id}_{token}"
                if token_key not in data[address]["tokenChangeLogs"]:
                    data[address]["tokenChangeLogs"][token_key] = {}
                data[address]["tokenChangeLogs"][token_key][str(timestamp)]={"amount": value.get('token_balance', 0)}
            result = list(data.values())
            self.exporter.update_documents("multichain_wallets", result)
            logger.info(f"Export 100 {self.chain_id} wallets in {time.time()-begin}s")