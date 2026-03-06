import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def heatMap(data):
    plt.figure()
    corr = data.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()
    
    
if __name__ == "__main__":
    data = pd.read_csv("ai_model_experiments_5000.csv")
    data.info()
    heatMap(data)