import pandas as pd

file_path = "Lending-Data-Ethereum.csv"
df = pd.read_csv(file_path)


def add_label(df):

    for index, row in df.iterrows():
        totalAsset = row["totalAsset"]
        numberOfLiquidation = row["numberOfLiquidation"]
        borrowInUSD = row["borrowInUSD"]
        depositInUSD = row["depositInUSD"]

        # Label 4

        if (
            totalAsset > 1000000
            and numberOfLiquidation == 0
            and borrowInUSD / totalAsset < 0.2
            # # and borrowInUSD > 0
            # and depositInUSD > 0
        ):
            df.loc[index, "1st_label"] = 4
            df.loc[index, "2nd_label"] = 4
        elif (
            totalAsset > 100000
            and numberOfLiquidation == 0
            and borrowInUSD == 0
            and depositInUSD > 0
        ):
            df.loc[index, "1st_label"] = 4
            df.loc[index, "2nd_label"] = 3

        # Label 3

        elif (
            totalAsset > 100000
            and numberOfLiquidation == 0
            and 0.2 <= borrowInUSD / totalAsset < 0.3
        ):
            df.loc[index, "1st_label"] = 3
            df.loc[index, "2nd_label"] = 4

        elif (
            10000 <= totalAsset < 100000
            and 0 < borrowInUSD / totalAsset < 0.2
            and numberOfLiquidation == 0
        ):
            df.loc[index, "1st_label"] = 3
            df.loc[index, "2nd_label"] = 3

        # Label 2

        elif 1000 < totalAsset < 10000 and numberOfLiquidation == 0:
            df.loc[index, "1st_label"] = 2
            df.loc[index, "2nd_label"] = 2

        elif (
            totalAsset > 10000
            and 0.3 <= borrowInUSD / totalAsset < 0.4
            and numberOfLiquidation == 0
        ):

            df.loc[index, "1st_label"] = 2
            df.loc[index, "2nd_label"] = 3

        # Label 1

        elif (
            totalAsset <= 1000
            and 0 <= numberOfLiquidation <= 3
            and depositInUSD > 0
            and borrowInUSD / totalAsset < 0.4
        ):
            df.loc[index, "1st_label"] = 1
            df.loc[index, "2nd_label"] = 1

        elif totalAsset <= 1000 and 0 <= numberOfLiquidation <= 3 and depositInUSD == 0:
            df.loc[index, "1st_label"] = 1
            df.loc[index, "2nd_label"] = 0

        # if (

        #     1000 < totalAsset <= 100000

        #     and 0.4 <= borrowInUSD / totalAsset < 0.6

        #     and numberOfLiquidation == 1

        # ):

        #     if (

        #         df.loc[index, "1st_label"] is not None

        #         and df.loc[index, "2nd_label"] is not None

        #     ):

        #         tmp = make_dict(

        #             df.loc[index, "address"],

        #             df.loc[index, "1st_label"],

        #             df.loc[index, "2nd_label"],

        #             1,

        #             2,

        #         )

        #         result_list.append(tmp)

        #     df.loc[index, "1st_label"] = 1

        #     df.loc[index, "2nd_label"] = 2

        elif (
            totalAsset > 1000
            and numberOfLiquidation > 0
            and 0 < borrowInUSD / totalAsset < 0.4
        ):
            df.loc[index, "1st_label"] = 1
            df.loc[index, "2nd_label"] = 0

        # Label 0

        elif totalAsset <= 1000 and numberOfLiquidation > 3:
            df.loc[index, "1st_label"] = 0
            df.loc[index, "2nd_label"] = 0
        elif (
            totalAsset > 0
            and 0.4 <= borrowInUSD / totalAsset < 0.6
            and numberOfLiquidation > 0
        ):

            df.loc[index, "1st_label"] = 0

            df.loc[index, "2nd_label"] = 1

        elif (
            totalAsset <= 1000
            and 0 <= numberOfLiquidation <= 3
            and depositInUSD > 0
            and borrowInUSD / totalAsset > 0.4
        ):

            df.loc[index, "1st_label"] = 1

            df.loc[index, "2nd_label"] = 0

        elif (
            1000 < totalAsset <= 100000
            and borrowInUSD / totalAsset >= 0.6
            and numberOfLiquidation > 0
        ):

            df.loc[index, "1st_label"] = 0
            df.loc[index, "2nd_label"] = 1

    df.to_csv("Lending-Data-Ethereum-Labeled.csv", index=False)

add_label(df)


