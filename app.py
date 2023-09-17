import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from eda import *

save_dataset()
df = open_dataset()

# Этот код не работает
st.pyplot(graphs_causality())


# А этот код работает
# fig, axes = plt.subplots(3, 2, figsize=(12, 12))
# fig.subplots_adjust(hspace=0.33)
#
# # 1. target vs. age
# sns.boxplot(data=df, y='AGE', x='TARGET', palette='Set1', ax=axes[0, 0])
# axes[0, 0].set_title('1. Зависимость возраста от таргета (BOXPLOT)', fontsize=11)
# axes[0, 0].set_xlabel('Таргет', fontsize=10)
# axes[0, 0].set_ylabel('Возраст', fontsize=10)
#
# sns.histplot(data=df, x='AGE', hue='TARGET', stat='percent', element='step', common_norm=False, ax=axes[0, 1])
# axes[0, 1].set_title('2. Зависимость возраста от таргета (ГИСТОГРАММА)', fontsize=11)
# axes[0, 1].set_xlabel('Возраст', fontsize=10)
# axes[0, 1].set_ylabel('Доля в %', fontsize=10)
#
# # 2. target vs. log of income
# sns.boxplot(data=df, y=np.log(df['PERSONAL_INCOME']), x='TARGET', palette='Set1', ax=axes[1, 0])
# axes[1, 0].set_title('3. Логарифм дохода от таргета (BOXPLOT)', fontsize=11)
# axes[1, 0].set_xlabel('Таргет', fontsize=10)
# axes[1, 0].set_ylabel('Логарифм дохода', fontsize=10)
#
# sns.histplot(data=df, x=np.log(df['PERSONAL_INCOME']), hue='TARGET', stat='percent', element='step', common_norm=False,
#              ax=axes[1, 1]);
# axes[1, 1].set_title('4. Логарифм дохода от таргета (ГИСТОГРАММА)', fontsize=11)
# axes[1, 1].set_xlabel('Логарифм дохода', fontsize=10)
# axes[1, 1].set_ylabel('Доля в %', fontsize=10)
# axes[1, 1].set_xlim(7.5, 11.5)
#
# # 3. target vs. share of closed loans
# sns.histplot(data=df, x='LOAN_CLOSED_SHARE', hue='TARGET', stat='percent', element='step',
#              common_norm=False, ax=axes[2, 0])
# axes[2, 0].set_title('5. Доля погашенных кредитов от таргета', fontsize=11)
# axes[2, 0].set_xlabel('Доля погашенных кредитов', fontsize=10)
# axes[2, 0].set_ylabel('Доля в таргете, %', fontsize=10)
#
# # 4. target vs. number of dependants
# sns.histplot(data=df, x='DEPENDANTS', hue='TARGET', stat='percent', element='step', binwidth=0.5,
#              common_norm=False, ax=axes[2, 1])
# axes[2, 1].set_title('6. Зависимость числа Dependants и таргета', fontsize=11)
# axes[2, 1].set_xlabel('Число Dependants', fontsize=10)
# axes[2, 1].set_ylabel('Доля в таргете, %', fontsize=10);
# st.pyplot(fig)

