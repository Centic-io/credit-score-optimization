import json

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.orm import sessionmaker

Session = sessionmaker()
engine = create_engine("postgresql://centic_write:centic2themoon@34.126.75.56:5432/postgres")
Session.configure(bind=engine)

with open('src/data/debtors.json', 'r') as f:
    debtors = json.loads(f.read())
chains = {
    "bnb": "0x38",
    "ethereum": "0x1",
    "polygon": "0x89",
    "arbitrum": "0xa4b1"
}
for idx in range(0, len(debtors), 1000):
    _wallets = ', '.join([f"'{wallet}'" for wallet in debtors[idx:idx+1000]])
    for key, value in chains.items():
        query = f"""
        SELECT * FROM chain_{value}.token_transfer
        WHERE from_address in ({_wallets})
        or to_address in ({_wallets})
        """
        result = []
        with Session.begin() as session:
            wallet_info = session.execute(query).all()
        for item in wallet_info:
            result.append({
                "contract_address": item.contract_address,
                "from_address": item.from_address,
                "to_address": item.to_address,
                "transaction_hash": item.transaction_hash,
                "log_index": item.log_index,
                "block_number": item.block_number,
                "value": item.value
            })
        with open(f"src/data/transfer_events/{key}/{idx}.json", "w") as f:
            json.dump(result,f, indent=1)