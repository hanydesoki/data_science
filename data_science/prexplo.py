import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def describe_columns(df):
    desc_df = {'Type':[],
              'NaN count':[],
              'NaN frequency':[],
              'Number of unique values':[]}
    for col in df.columns:
        desc_df['Type'].append(df[col].dtype)
        desc_df['NaN count'].append(pd.isnull(df[col]).sum())
        desc_df['NaN frequency'].append(pd.isnull(df[col]).mean())
        desc_df['Number of unique values'].append(len(df[col].unique()))
    return pd.DataFrame(desc_df, index=df.columns)

def move_column(df, column_name, column_place):
    mvd_column = df.pop(column_name)
    df.insert(column_place, column_name, mvd_column)
    return df

def prop_nan(df):
    return (df.isna()).sum().sum()/df.size

def nan_map(df, figsize=(20,10), save=False, filename='nan_location'):
    plt.figure(figsize=figsize)
    sns.heatmap(df.isna())
    if save:
        plt.savefig(filename)