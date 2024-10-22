import json
import os

from defi_services.constants.query_constant import Query
from defi_services.jobs.processors.multi_call_state_processor import MultiCallStateProcessor
from dotenv import load_dotenv

from src.backtest.cli_job import CLIJob
from src.backtest.database.klg_mongodb import MongoDbKLG
from src.backtest.database.memory_storage import MemoryStorage
from src.backtest.database.mongodb import MongoDB
from src.constants.constants_v3 import Chain
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
            for key, value in wallet.get("tokens").items():
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
            self._prepare_queries(keys[idx:idx+100])
            response_data = self.multicall.run(self.queries, batch_size=2000, max_workers=1, ignore_error=True)
            for key, value in response_data:
                pass

if __name__ == "__main__":
    with open("balance_timestamp.json", "r") as f:
        wallets = json.loads(f.read())
    for chain in ["0x38", "0x1", "0x89", "0xa4b1"]:
        wallet_timestamps = {key: value[chain] for key, value in wallets.items() }
        prefix = Chain.prefix.get(chain)
        _importer = MongoDB(os.environ.get("MONGO_MAIN"), "blockchain_etl", prefix)
        _exporter = MongoDB("mongodb://localhost:27017/", "knowledge_graph")
        _klg_db = MongoDbKLG(os.environ.get("ENTITIES_DB"), chain_id=chain)
        _protocol_db = MongoDB(os.environ.get("DAPP_INFO_DB"), "SmartContractLabel")
        if not prefix:
            prefix = "bsc"
        provider = os.environ.get(f"{prefix.upper()}_PROVIDER")
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
