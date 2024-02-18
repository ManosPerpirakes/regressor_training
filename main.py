import pandas as pd

def set_data(df1):
    df = pd.read_csv(df1)
    df = df.drop([
        "MSZoning",
        "Street",
        "Alley",
        "LotShape",
        "LandContour",
        "Utilities",
        "LotConfig",
        "LandSlope",
        "Neighborhood",
        "Condition1",
        "Condition2",
        "BldgType",
        "HouseStyle",
        "RoofStyle",
        "RoofMatl",
        "Exterior1st",
        "Exterior2nd",
        "MasVnrType",
        "ExterQual",
        "ExterCond",
        "Foundation",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "Heating",
        "HeatingQC",
        "CentralAir",
        "Electrical",
        "KitchenQual",
        "Functional",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "PavedDrive",
        "PoolQC",
        "Fence",
        "MiscFeature",
        "SaleType",
        "SaleCondition"
        ], 
        axis=1
    )
    df.fillna(1, inplace=True)
    print(df.head())
    print(df.info())
    return df


df = set_data("train.csv")
df_test = set_data("test.csv")
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X)

from sklearn.neighbors import KNeighborsRegressor

classiefier = KNeighborsRegressor(n_neighbors=5)
classiefier.fit(X, y)
y_pred = classiefier.predict(df_test)

result = pd.DataFrame({"SalePrice": y_pred})
counter = 1460
def set_id(var):
    global counter
    counter += 1
    return counter
result["Id"] = pd.Series()
result["Id"] = result["Id"].apply(set_id)
result.to_csv("results.csv", index=False)