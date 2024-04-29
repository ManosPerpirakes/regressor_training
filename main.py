import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

def set_data(df1):
    df = pd.read_csv(df1)
    df[list(pd.get_dummies(df['Neighborhood']).columns)] = pd.get_dummies(df['Neighborhood'])
    df[list(pd.get_dummies(df['SaleType']).columns)] = pd.get_dummies(df['SaleType'])
    df[list(pd.get_dummies(df['SaleCondition']).columns)] = pd.get_dummies(df['SaleCondition'])
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
    return df


df = set_data("train.csv")
df_test = set_data("test.csv")
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']
sc = StandardScaler()
X_train = sc.fit_transform(X)
regressor = KNeighborsRegressor(n_neighbors=7)
regressor.fit(X, y)
y_pred = regressor.predict(df_test)
lst = []
for i in range(1461, 2920):
    lst.append(i)
result = pd.DataFrame({"Id": lst})
result["SalePrice"] = y_pred
result.to_csv("results.csv", index=False)
print("DONE")