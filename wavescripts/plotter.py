#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 16:27:38 2025

@author: gpt
"""

import matplotlib.pyplot as plt

def plot_second_column(df,rangestart,rangeend, title=None):
    """
    Plots the 2nd column of a pandas DataFrame.
    """
    df.iloc[rangestart:rangeend, 1].plot()
    if title:
        plt.title(title)
    plt.xlabel(df.columns[1])
    plt.show()



def plot_column(df,rangestart, rangeend, col_index=1, title=None):
    """
    More flexible version: plot any column by index.
    """
    df.iloc[rangestart:rangeend, col_index].plot()
    if title:
        plt.title(title)
    plt.xlabel(df.columns[col_index])
    plt.show()
