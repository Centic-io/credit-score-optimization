import pandas as pd

class DataPreprocessor:
    def __init__(self, path_file):
        self.df = pd.read_csv(path_file)

    def add_label_excellent(self, total_asset, number_of_liquidation, borrow_in_usd, deposit_in_usd, index):
        if (
            total_asset > 1000000
            and not number_of_liquidation
            and borrow_in_usd / total_asset <= 0.2
        ):
            self.df.loc[index, "1st_label"] = 4
            self.df.loc[index, "2nd_label"] = 4

        if (
            total_asset > 100000
            and not number_of_liquidation
            and not borrow_in_usd
            and deposit_in_usd
        ) or (
            10000 < total_asset <= 100000
            and 0.2 >= borrow_in_usd / total_asset > 0
            and not number_of_liquidation
        ):
            self.df.loc[index, "1st_label"] = 3
            self.df.loc[index, "2nd_label"] = 4

        return self.df

    def add_label_very_good(self, total_asset, number_of_liquidation, borrow_in_usd, index):
        if (
            total_asset > 1000000
            and not number_of_liquidation
            and 0.2 < borrow_in_usd / total_asset <= 0.3
        ):
            self.df.loc[index, "1st_label"] = 3
            self.df.loc[index, "2nd_label"] = 3

        if (
            1000 < total_asset <= 10000
            and not number_of_liquidation
        ):
            self.df.loc[index, "1st_label"] = 2
            self.df.loc[index, "2nd_label"] = 3

        return self.df

    def add_label_good(self, total_asset, number_of_liquidation, borrow_in_usd, index):
        if (
            total_asset > 1000
            and not number_of_liquidation
            and 0.3 < borrow_in_usd / total_asset <= 0.4
        ):
            self.df.loc[index, "1st_label"] = 2
            self.df.loc[index, "2nd_label"] = 2

        if (
            0 < total_asset <= 100000
            and not number_of_liquidation
            and 0.4 < borrow_in_usd / total_asset <= 0.6
        ):
            self.df.loc[index, "1st_label"] = 1
            self.df.loc[index, "2nd_label"] = 2

        return self.df

    def add_label_fair(self, total_asset, number_of_liquidation, borrow_in_usd, deposit_in_usd, index):
        if (
            total_asset <= 1000
            and number_of_liquidation <= 3
        ):
            self.df.loc[index, "1st_label"] = 1
            self.df.loc[index, "2nd_label"] = 1

        if (
            number_of_liquidation > 0
            and 0.4 < borrow_in_usd / total_asset
        ) or (
            0 < total_asset <= 100000
            and borrow_in_usd / total_asset > 0.6
            and number_of_liquidation > 0
        ) or (
            total_asset > 1000
            and borrow_in_usd / total_asset <= 0.4
            and number_of_liquidation > 0
        ) or (
            0 < total_asset <= 1000
            and not deposit_in_usd
            and 0 < number_of_liquidation <= 3
        ):
            self.df.loc[index, "1st_label"] = 0
            self.df.loc[index, "2nd_label"] = 1

        return self.df

    def add_label_poor(self, total_asset, number_of_liquidation, index):
        if (
            0 < total_asset <= 1000
            and number_of_liquidation > 3
        ):
            self.df.loc[index, "1st_label"] = 0
            self.df.loc[index, "2nd_label"] = 0

        return self.df

    def add_label(self):
        for index, row in self.df.iterrows():
            total_asset = row["totalAsset"]
            number_of_liquidation = row["numberOfLiquidation"]
            borrow_in_usd = row["borrowInUSD"]
            deposit_in_usd= row["depositInUSD"]
            self.add_label_excellent(total_asset, number_of_liquidation, borrow_in_usd, deposit_in_usd, index)
            self.add_label_very_good(total_asset, number_of_liquidation, borrow_in_usd, index)
            self.add_label_good(total_asset, number_of_liquidation, borrow_in_usd, index)
            self.add_label_fair(total_asset, number_of_liquidation, borrow_in_usd, deposit_in_usd, index)
            self.add_label_poor(total_asset, number_of_liquidation, index)

        self.df.to_csv("Lending-Data-Ethereum-Labeled.csv", index=False)


if __name__ == '__main__':
    job = DataPreprocessor('Lending-Data-Ethereum.csv')
    job.add_label()