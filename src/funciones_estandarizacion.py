from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, RobustScaler

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

def estandarizacion(df, col_num, modelos_estand):

    for estand in modelos_estand:

        if estand == "RobustScaler":
            escalador = RobustScaler()
            colums_escaladas = [f"{elem}_robust" for elem in col_num]
            
        elif estand == "StandardScaler":
            escalador = StandardScaler()
            colums_escaladas = [f"{elem}_standar" for elem in col_num]
            
        elif estand == "MinMaxScaler":
            escalador = MinMaxScaler()
            colums_escaladas = [f"{elem}_minmax" for elem in col_num]

        elif estand == "Normalizer":
            escalador = Normalizer()
            colums_escaladas = [f"{elem}_normalizer" for elem in col_num]
            
        datos_transf = escalador.fit_transform(df[col_num])
        df[colums_escaladas] = datos_transf

    return df


def visualizacion_estandarizacion(data, columnas, num_columnas, figsize=(15,10)):

    num_filas = math.ceil(len(columnas) / num_columnas)

    fig, axes = plt.subplots(nrows=num_filas, ncols=num_columnas, figsize=figsize)
    axes = axes.flat

    for i, col in enumerate(columnas):
        sns.boxplot(x=col, data=data, ax=axes[i])
        axes[i].set_title(f"Boxplot de {col}")

    for j in range(len(columnas), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()