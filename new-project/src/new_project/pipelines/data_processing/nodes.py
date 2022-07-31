"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.2
"""
import numpy as np
import pandas as pd 
import featuretools as ft
from category_encoders import TargetEncoder


def cleaning_application_train(data):

    # data.drop(['SK_ID_CURR','CODE_GENDER'],axis=1,inplace=True)
    data[data['AMT_INCOME_TOTAL'] >40000000]['AMT_INCOME_TOTAL']=np.nan
    data[data['DAYS_EMPLOYED']>100000]['DAYS_EMPLOYED']=np.nan
    data[data['OWN_CAR_AGE']>50]['OWN_CAR_AGE']=np.nan
    data[data['OBS_30_CNT_SOCIAL_CIRCLE']>340]['OBS_30_CNT_SOCIAL_CIRCLE']=np.nan
    data[data['OBS_60_CNT_SOCIAL_CIRCLE']>340]['OBS_60_CNT_SOCIAL_CIRCLE']=np.nan


    return data


def cleaning_bureau(data):
    data[data['DAYS_ENDDATE_FACT']<-40000]['DAYS_ENDDATE_FACT']=np.nan
    data[data['DAYS_CREDIT_UPDATE']<-40000]['DAYS_CREDIT_UPDATE']=np.nan
    data[data['AMT_CREDIT_MAX_OVERDUE']>.8e8]['AMT_CREDIT_MAX_OVERDUE']=np.nan
    data[data['AMT_CREDIT_SUM']>5e8]['AMT_CREDIT_SUM']=np.nan
    data[data['AMT_CREDIT_SUM_DEBT']>1.5e8]['AMT_CREDIT_SUM_DEBT']=np.nan 
    data[data['AMT_CREDIT_SUM_OVERDUE']>3.5e5]['AMT_CREDIT_SUM_OVERDUE']=np.nan 
    data[data['AMT_ANNUITY']>1e8]['AMT_ANNUITY']=np.nan

    return data



def cleaning_previous_application(data):
    data['DAYS_FIRST_DRAWING'][data['DAYS_FIRST_DRAWING'] == 365243.0] = np.nan
    data['DAYS_FIRST_DUE'][data['DAYS_FIRST_DUE'] == 365243.0] = np.nan
    data['DAYS_LAST_DUE_1ST_VERSION'][data['DAYS_LAST_DUE_1ST_VERSION'] == 365243.0] = np.nan
    data['DAYS_LAST_DUE'][data['DAYS_LAST_DUE'] == 365243.0] = np.nan
    data['DAYS_TERMINATION'][data['DAYS_TERMINATION'] == 365243.0] = np.nan
    data['AMT_APPLICATION'][data['AMT_APPLICATION'] > 5e6] = np.nan

    return data


def feature_engineering_application_train(data):

    data.drop('CODE_GENDER',axis=1,inplace=True)

    data['EXT_SOURCE']=data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    data.drop(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'],axis=1,inplace=True)
    negative_days=['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE']
    for i in negative_days:
        data[i]=np.abs(data[i])

    data['DAYS_BIRTH']=np.round(data['DAYS_BIRTH']/365)

    data['DAYS_EMPLOYED']=np.round(data['DAYS_EMPLOYED']/365)

    data["PAYMENT_RATE"]=data["AMT_ANNUITY"]/data["AMT_CREDIT"]

    data["INCOME_PER_PERSON"]=data["AMT_INCOME_TOTAL"]/data["AMT_CREDIT"]

    data["INCOME_PER_PERSON_RATE"]=data["AMT_INCOME_TOTAL"]/(data["CNT_FAM_MEMBERS"]+1)

    data["DAYS_WORKING_PER"]=data["DAYS_EMPLOYED"]/data["DAYS_BIRTH"]

    data["ANNUITY_DAYS_EMPLOYED_PER"]=data["DAYS_EMPLOYED"]/data["AMT_ANNUITY"]

    data["AMT_CREDIT_DAYS_EMPLOYED"]=data["DAYS_EMPLOYED"]/data["AMT_CREDIT"]

    data['GOODS_PRICE_AFFORDABLE']=data['AMT_INCOME_TOTAL']/data['AMT_GOODS_PRICE']



    return data

def feature_engineering_bureau(data):
    
    data['ANNUITY_CREDIT_RATIO'] = data['AMT_ANNUITY'] / data['AMT_CREDIT_SUM']


    return data

def feature_engineering_bureau_balance(data):
    
    data["MONTHS_BALANCE"]=np.abs(data["MONTHS_BALANCE"])
    data['STATUS']=data['STATUS'].apply(lambda x: 'Y' if x in ['5', '1', '4', '3','0', '2'] else x)

    return data

def feature_engineering_POS_CASH_balance(data):
    
    data["MONTHS_BALANCE"]=np.abs(data["MONTHS_BALANCE"])
    data['SK_DPD'] = data['SK_DPD'].apply( lambda x: 1 if x>0 else 0)

    return data

def feature_engineering_credit_card_balance(data):
    
    data["MONTHS_BALANCE"]=np.abs(data["MONTHS_BALANCE"])
    data['SK_DPD'] = data['SK_DPD'].apply( lambda x: 1 if x>0 else 0)

    return data


def feature_engineering_installments_payments(data):
    
   data['LATE_PAYMENT_FLAG'] = (data['DAYS_ENTRY_PAYMENT'] - data['DAYS_INSTALMENT']).apply(lambda x: 1 if x>0 else 0)
   data['LESS_PAYMENT_FLAG'] = (data['AMT_PAYMENT'] - data['AMT_INSTALMENT']).apply(lambda x: 1 if x>0 else 0)

   return data


def feature_engineering_previous_application(data):
    
   data["PAYMENT_RATE"]=data["AMT_ANNUITY"]/data["AMT_CREDIT"]

   return data



#Nodes

def preprocess_installments_payments(installments_payments: pd.DataFrame) -> pd.DataFrame:
    installments_payments=feature_engineering_installments_payments(installments_payments)
    return installments_payments


def preprocess_application_train(application_train: pd.DataFrame) -> pd.DataFrame:
    application_train=cleaning_application_train(application_train)
    application_train=feature_engineering_application_train(application_train)

    return application_train


def preprocess_previous_application(previous_application: pd.DataFrame) -> pd.DataFrame:
    previous_application=feature_engineering_previous_application(previous_application)
    previous_application=cleaning_previous_application(previous_application)
    return previous_application



def preprocess_credit_card_balance(credit_card_balance: pd.DataFrame) -> pd.DataFrame:
    credit_card_balance=feature_engineering_credit_card_balance(credit_card_balance)
    return credit_card_balance


def preprocess_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    bureau=cleaning_bureau(bureau)
    bureau=feature_engineering_bureau(bureau)
    return bureau


def preprocess_bureau_balance(bureau_balance: pd.DataFrame) -> pd.DataFrame:
    bureau_balance=feature_engineering_bureau_balance(bureau_balance)
    return bureau_balance


def preprocess_pos_cash_balance(pos_cash_balance: pd.DataFrame) -> pd.DataFrame:
    pos_cash_balance=feature_engineering_POS_CASH_balance(pos_cash_balance)
    return pos_cash_balance




def merging_tables(bureau,bureau_balance,application_train,previous_application,credit_card_balance,installments_payments,pos_cash_balance):
    es = ft.EntitySet(id="application")
    es = es.add_dataframe(dataframe_name="bureau",
                                dataframe=bureau,
                                index="SK_ID_BUREAU")

    es = es.add_dataframe(dataframe_name="bureau_balance",
                                dataframe=bureau_balance,
                                index="index")

    es = es.add_dataframe(dataframe_name="application_train",
                                dataframe=application_train,
                                index="SK_ID_CURR")

    es = es.add_dataframe(dataframe_name="previous_application",
                                dataframe=previous_application,
                                index="SK_ID_PREV")

    es = es.add_dataframe(dataframe_name="pos_cash_balance",
                                dataframe=pos_cash_balance,
                                index="index")  

    es = es.add_dataframe(dataframe_name="installments_payments",
                                dataframe=installments_payments,
                                index="index")

    es = es.add_dataframe(dataframe_name="credit_card_balance",
                                dataframe=credit_card_balance,
                                index="index")    
    es = es.add_relationship("bureau", "SK_ID_BUREAU", "bureau_balance", "SK_ID_BUREAU")
    es = es.add_relationship("application_train", "SK_ID_CURR", "bureau", "SK_ID_CURR")
    
    es = es.add_relationship("application_train", "SK_ID_CURR", "previous_application", "SK_ID_CURR")
    es = es.add_relationship("previous_application", "SK_ID_PREV", "installments_payments", "SK_ID_PREV")
    es = es.add_relationship("previous_application", "SK_ID_PREV", "credit_card_balance", "SK_ID_PREV")
    es = es.add_relationship("previous_application", "SK_ID_PREV", "pos_cash_balance", "SK_ID_PREV")

    feature_matrix, feature_defs =  ft.dfs(entityset=es, target_dataframe_name="application_train")
    return feature_matrix

def encoding(data,liste):
    for i in liste:
        model=TargetEncoder(cols=[i])
        data[i]=model.fit_transform(data[i],data['TARGET'])
    return data


def create_model_input_table(
    pos_cash_balance: pd.DataFrame, bureau_balance: pd.DataFrame, bureau: pd.DataFrame, credit_card_balance: pd.DataFrame, 
    previous_application: pd.DataFrame,installments_payments: pd.DataFrame,application_train: pd.DataFrame
) -> pd.DataFrame:
    data=merging_tables(bureau,bureau_balance,application_train,previous_application,credit_card_balance,installments_payments,pos_cash_balance)
    categorical_features=data.select_dtypes('O').columns
    numerical_features=data.select_dtypes(exclude="O").columns
    for i in categorical_features:
        data[i]=data[i].fillna(data[i].mode()[0]) 
    for i in numerical_features:
            if(data[i].mean()==np.nan):
                data[i]=data[i].fillna(data[i].mean())
            else : 
                data[i]=data[i].fillna(0)
    infPos=[]
    infNeg=[]
    for i in numerical_features:
        if(data[i].max()==np.inf):
            infPos.append(i)
        elif (data[i].min()==-np.inf):
            infNeg.append(i)
    for i in infPos:
        data[i]=data[i].replace(np.inf,np.nan)
        data[i]=data[i].replace(np.nan,data[i].max())
    for i in infNeg:
        data[i]=data[i].replace(-np.inf,np.nan)
        data[i]=data[i].replace(np.nan,data[i].min())
    
    data=encoding(data,categorical_features)

    cor_matrix = data.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    data.drop(to_drop,axis=1,inplace=True)

    return data


