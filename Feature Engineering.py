#####################################################################################
#                                  Feature Engineering                              #
#####################################################################################

#GEREKLİ KÜTÜPHANELER
import argparse
from sklearn.preprocessing import StandardScaler
from helpers.eda import *
from helpers.data_prep import *
import pandas as pd
import numpy as np
! pip install missingno
import missingno as msno



#VERİNİN YÜKLENMESİ
def titanic_load():
    data = pd.read_csv("titanic.csv")
    data.columns = [col.upper() for col in data.columns]
    return data

df = titanic_load()
#GÖREV1:Çalışma dizininde helpers adında bir dizin açıp, içerisine data_prep.py adında bir script ekleyiniz.
# Feature Engineering bölümünde kendimize ait tüm  fonksiyonları, bu script içerisine toplayınız.


def titanic_feature_engineering(dataframe):
    # Cabin bool
    dataframe["NEW_CABIN_BOOL"] = dataframe["CABIN"].notnull().astype('int')
    # Name count
    dataframe["NEW_NAME_COUNT"] = dataframe["NAME"].str.len()
    # name word count
    dataframe["NEW_NAME_WORD_COUNT"] = dataframe["NAME"].apply(lambda x: len(str(x).split(" ")))
    # name dr
    dataframe["NEW_NAME_DR"] = dataframe["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
    # name title
    dataframe['NEW_TITLE'] = dataframe.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
    # family size
    dataframe["NEW_FAMILY_SIZE"] = dataframe["SIBSP"] + dataframe["PARCH"] + 1
    # age_pclass
    dataframe["NEW_AGE_PCLASS"] = dataframe["AGE"] * dataframe["PCLASS"]
    # is alone
    dataframe.loc[((dataframe['SIBSP'] + dataframe['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
    dataframe.loc[((dataframe['SIBSP'] + dataframe['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
    # age level
    dataframe.loc[(dataframe['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    dataframe.loc[(dataframe['AGE'] >= 18) & (dataframe['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    dataframe.loc[(dataframe['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
    # sex x age
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (
            (dataframe['AGE'] > 21) & (dataframe['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (
            (dataframe['AGE'] > 21) & (dataframe['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

    print(dataframe.shape)
    dataframe.columns = [col.upper() for col in dataframe.columns]
    return dataframe


def titanic_missing_values(dataframe):
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    num_cols = [col for col in num_cols if "PASSENGERID" not in col]

    for col in num_cols:
        print(col, check_outlier(dataframe, col))
    for col in num_cols:
        replace_with_thresholds(dataframe, col)
    for col in num_cols:
        print(col, check_outlier(dataframe, col))

    missing_values_table(dataframe, True)
    remove_cols = ["TICKET", "NAME", "CABIN"]
    dataframe.drop(remove_cols, inplace=True, axis=1)
    dataframe["AGE"] = dataframe["AGE"].fillna(dataframe.groupby("NEW_TITLE")["AGE"].transform("median"))
    dataframe["NEW_AGE_PCLASS"] = dataframe["AGE"] * dataframe["PCLASS"]
    dataframe.loc[(dataframe['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    dataframe.loc[(dataframe['AGE'] >= 18) & (dataframe['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    dataframe.loc[(dataframe['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    dataframe.loc[(dataframe['SEX'] == 'male') & ((dataframe['AGE'] > 21) & (dataframe['AGE']) < 50), 'NEW_SEX_CAT'] \
        = 'maturemale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & ((dataframe['AGE'] > 21) & (dataframe['AGE']) < 50), 'NEW_SEX_CAT'] \
        = 'maturefemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

    missing_values_table(dataframe)
    dataframe = dataframe.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and
                                                                    len(x.unique()) <= 10) else x, axis=0)
    dataframe.columns = [col.upper() for col in dataframe.columns]
    return dataframe


def titanic_encoding(dataframe):
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    num_cols = [col for col in num_cols if "PASSENGERID" not in col]

    # Label Encoding
    binary_cols = [col for col in dataframe.columns if dataframe[col].dtype not in [int, float]
                   and dataframe[col].nunique() == 2]
    for col in binary_cols:
        dataframe = label_encoder(dataframe, col)

    # Rare Encoding
    rare_analyser(dataframe, "SURVIVED", cat_cols)
    dataframe = rare_encoder(dataframe, 0.01)
    rare_analyser(dataframe, "SURVIVED", cat_cols)

    # One-Hot Encoding
    ohe_cols = [col for col in dataframe.columns if 10 >= dataframe[col].nunique() > 2]
    dataframe = one_hot_encoder(dataframe, ohe_cols)

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    rare_analyser(dataframe, "SURVIVED", cat_cols)
    dataframe = rare_encoder(dataframe, 0.01)
    rare_analyser(dataframe, "SURVIVED", cat_cols)

    dataframe.columns = [col.upper() for col in dataframe.columns]
    return dataframe


def titanic_standardization(dataframe):
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    num_cols = [col for col in num_cols if "PASSENGERID" not in col]

    scaler = StandardScaler()
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])
    return dataframe


#GÖREV2:titanic_data_prep adında bir fonksiyon yazınız.

def titanic_data_prep():
    dataframe = titanic_load()
    check_dataframe(dataframe)
    dataframe = titanic_feature_engineering(dataframe)
    dataframe = titanic_missing_values(dataframe)
    dataframe = titanic_encoding(dataframe)
    dataframe = titanic_standardization(dataframe)
    check_dataframe(dataframe)
    return dataframe


check_df(df)


#GÖREV 3:Veri ön işleme yaptığınız veri setini pickle ile diske kaydediniz.

df.to_pickle("prepared_titanic_df.pkl")

dff = pd.read_pickle("prepared_titanic_df.pkl")


