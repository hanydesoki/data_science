import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def boxplot_groupes(df, categ_column, target_column, figsize=(20, 10)):
    groupes = []
    for cat in list(df[categ_column].unique()):
        groupes.append(df[df[categ_column] == cat][target_column])

    medianprops = {'color': "black"}
    meanprops = {'marker': 'o', 'markeredgecolor': 'black',
                 'markerfacecolor': 'firebrick'}

    plt.figure(figsize=figsize)
    plt.boxplot(groupes, labels=list(df[categ_column].unique()), showfliers=False, medianprops=medianprops,
                vert=False, patch_artist=True, showmeans=True, meanprops=meanprops)
    plt.ylabel(categ_column)
    plt.title(target_column)


def plot_map(df, y_name, figsize=(12, 10), lat_long=('Long', 'Lat'), boundaries=None, interquartile=True, alpha=1):
    plt.figure(figsize=figsize)
    desc_df = df.describe()
    if interquartile:
        vmin = desc_df.loc['25%', y_name]
        vmax = desc_df.loc['75%', y_name]
    else:
        vmin = desc_df.loc['min', y_name]
        vmax = desc_df.loc['max', y_name]

    points = plt.scatter(df[lat_long[0]], df[lat_long[1]], c=df[y_name], cmap='jet', lw=0, s=10, vmin=vmin, vmax=vmax,
                         alpha=alpha)
    if boundaries != None:
        plt.xlim(boundaries[0])
        plt.ylim(boundaries[1])
    plt.xlabel(lat_long[0])
    plt.ylabel(lat_long[1])
    plt.colorbar(points)


def plot_map_categ(df, categ_column, figsize=(12, 10), lat_long=('Long', 'Lat'), boundaries=None, alpha=1):
    plt.figure(figsize=figsize)
    for classe in df[categ_column].sort_values().unique():
        df_classe = df[df[categ_column] == classe]
        plt.scatter(df_classe[lat_long[0]], df_classe[lat_long[1]], lw=0, s=10, label=classe, alpha=alpha)
    if boundaries != None:
        plt.xlim(boundaries[0])
        plt.ylim(boundaries[1])
    plt.legend()


def corr_matrix(df, figsize=(30, 20), maptype='heatmap', absolute=False, crit_value=None,
                annot=True, save=False, filename='corr_matrix'):
    matrix_corr = df.corr()

    if absolute:
        matrix_corr = matrix_corr.abs()
    if crit_value != None:
        matrix_corr = matrix_corr >= crit_value
    plt.figure(figsize=figsize)
    if maptype == 'heatmap':
        sns.heatmap(matrix_corr, annot=annot)
    elif maptype == 'clustermap':
        sns.clustermap(matrix_corr, annot=annot)

    if save:
        plt.savefig(filename)