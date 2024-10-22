import json
import time
import os
from dotenv import load_dotenv
from abi.vtoken import VTOKEN_ABI
from abi.erc20_abi import ERC20_ABI
from database.klg_mongodb import MongoDbKLG
from database.mongodb import MongoDB
from database.memory_storage import MemoryStorage
from cli_job import CLIJob
from src.constants.constants_v3 import Amount, RemoveToken, CompoundForks, Chain
from src.utils.logger_utils import get_logger
from web3 import Web3, HTTPProvider
load_dotenv()
logger = get_logger("Liquidated Wallet")


class ExportLiquidatedWalletJob(CLIJob):
    def __init__(
            self,
            importer: MongoDB,
            exporter: MongoDB,
            klg_db: MongoDbKLG,
            protocol_db: MongoDB,
            wallets,
            batch_size,
            chain_id):
        super().__init__()
        self.protocol_name = {}
        self.wallets = wallets
        self.protocol_db = protocol_db
        self.ctoken_addresses = {}
        self.exchange_rate = {}
        self.underlying = {}
        self.klg_db = klg_db
        self.batch_size = batch_size
        self.exporter = exporter
        self.importer = importer
        self.chain_id = chain_id
        self.local_storage = MemoryStorage.get_instance()
        self.get_prepare_protocol_information()

    def _execute(self, *args, **kwargs):
        for idx in range(0, len(self.wallets), 1000):
            self._execute_liquidate_events(self.wallets[idx:idx+1000])

    def _execute_liquidate_events(self, addresses):
        logger.info("Start crawling...")
        cursor = self.importer.get_documents(
            collection="lending_events",
            conditions={
                "event_type": "LIQUIDATE",
                "user": {"$in": addresses}
            }
        )
        liquidators, debtors = [], []
        count = 0
        begin = time.time()
        for event in cursor:
            if "debt_asset" not in event:
                if event["contract_address"] not in self.underlying:
                    continue
                event["debt_asset"] = self.underlying[event["contract_address"]]
            if event["collateral_asset"] in self.underlying:
                event["collateral_asset"] = self.underlying[event["collateral_asset"]]

            if event["debt_asset"] in RemoveToken.tokens or event["collateral_asset"] in RemoveToken.tokens:
                continue

            amount_in_usd = {}
            for key in Amount.all:
                if key in event:
                    amount_in_usd[key] = event[key]
                elif key == Amount.liquidated_collateral_amount_in_usd:
                    amount_in_usd[key] = event[Amount.mapping[key]] * \
                                         self.exchange_rate.get(event['collateral_asset'], 1) * \
                                         self.get_token_price(event[Amount.token[key]], event["block_timestamp"])
                else:
                    amount_in_usd[key] = event[Amount.mapping[key]] * \
                                         self.get_token_price(event[Amount.token[key]], event["block_timestamp"])
            liquidator = {
                "_id": event.get("wallet") or event.get("liquidator"),
                "debtors": {
                    event["user"]: {
                        str(event["block_timestamp"]): {
                            "protocol": event["contract_address"],
                            "collateralAsset": event["collateral_asset"],
                            "collateralAmount": event["liquidated_collateral_amount"],
                            "collateralAssetInUSD": amount_in_usd[Amount.liquidated_collateral_amount_in_usd],
                            "debtor": event["user"],
                            "blockNumber": event["block_number"]
                        }
                    }
                }
            }
            debtor = {
                "_id": event["user"],
                "buyers": {
                    event.get("wallet") or event.get("liquidator"): {
                        str(event["block_timestamp"]): {
                            "protocol": event["contract_address"],
                            "debtAsset": event["debt_asset"],
                            "debtAmount": event["debt_to_cover"],
                            "debtAssetInUSD": amount_in_usd[Amount.debt_to_cover_in_usd],
                            "buyer": event.get("wallet") or event.get("liquidator"),
                            "blockNumber": event["block_number"]
                        }
                    }
                }
            }
            liquidators.append(liquidator)
            debtors.append(debtor)
            if len(liquidators) == self.batch_size:
                count += self.batch_size
                logger.info(f"Export {count} events in {time.time() - begin}s")
                self.exporter.update_documents("liquidators", liquidators)
                self.exporter.update_documents("debtors", debtors)
                liquidators, debtors = [], []

        count += len(liquidators)
        logger.info(f"Export {count} events in {time.time() - begin}s")
        self.exporter.update_documents("liquidators", liquidators)
        self.exporter.update_documents("debtors", debtors)

    def get_token_price(self, token, time_):
        key = f"{self.chain_id}_{token}"
        price = self.local_storage.get(key)
        if not price:
            price = self.klg_db.get_smart_contract(key)

        result = price.get("price")
        if "priceChangeLogs" in price and price.get("priceChangeLogs"):
            for timestamp in price["priceChangeLogs"]:
                result = price["priceChangeLogs"][timestamp]
                if int(timestamp) >= time_:
                    break
        if result is None:
            print(token)
            result = 1
        self.local_storage.set(key, price)

        return result

    def get_prepare_protocol_information(self):
        cursor = self.protocol_db.get_documents("protocols", {"_id": {"$regex": "aave"}, "chainId": self.chain_id})
        for item in cursor:
            self.protocol_name[item.get("address")] = "aave"
        cursor = self.protocol_db.get_documents("protocols", {"_id": {"$regex": "venus"}, "chainId": self.chain_id})
        self.get_underlying_ctoken(cursor, protocol_name="venus")
        cursor = self.protocol_db.get_documents("protocols", {"_id": {"$regex": "compound"}, "chainId": self.chain_id})
        self.get_underlying_ctoken(cursor, protocol_name="compound")

    def get_underlying_ctoken(self, cursor, protocol_name):
        for item in cursor:
            if "compound-v3" in item.get("_id"):
                continue
            if "morpho-compound" in item.get("_id"):
                continue
            for token in item["reservesList"]:
                ctoken = item['reservesList'][token]['cToken']
                self.underlying[ctoken] = token
                self.ctoken_addresses[token] = ctoken
                self.exchange_rate[token] = item['reservesList'][token]["exchangeRate"]
                self.protocol_name[ctoken] = protocol_name

if __name__ == "__main__":
    with open("debtor_type.json", "r") as f:
        wallets = json.loads(f.read())
    wallets = [key for key, value in wallets.items() if value == "wallet"]
    for chain in ["0x38", "0x1", "0x89", "0xa4b1"]:
        prefix = Chain.prefix.get(chain)
        _importer = MongoDB(os.environ.get("MONGO_MAIN"), "blockchain_etl", prefix)
        _exporter = MongoDB("mongodb://localhost:27017/", "blockchain_etl", prefix)
        _klg_db = MongoDbKLG(os.environ.get("ENTITIES_DB"), chain_id=chain)
        _protocol_db = MongoDB(os.environ.get("DAPP_INFO_DB"), "SmartContractLabel")
        job = ExportLiquidatedWalletJob(
            chain_id=chain,
            importer=_importer,
            exporter=_exporter,
            klg_db=_klg_db,
            protocol_db=_protocol_db,
            batch_size=1000,
            wallets=wallets
        )
        job.run()
