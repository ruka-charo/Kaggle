'''kaggle titanic'''
# どのような人が生き残りやすかったか
%cd /Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Kaggle/Titanic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder


'''データ前処理'''
#%% データの読み込み & 前処理
df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')
df_ans = pd.read_csv('./data/gender_submission.csv')

#%% データの取捨選択
df_train = df_train.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df_train['Embarked'] = df_train['Embarked'].fillna('S')
df_train.head()

#%% 欠損値の確認
df_train.isnull().sum()
df_train['Embarked']

#%% 年齢を平均値で埋め、欠損があったかどうかの新しい特徴量を作成する。欠損値あり:1
age_nan = df_train[df_train['Age'].isnull()]
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())
df_train['Age_NaN'] = 0
for i in age_nan.index:
    df_train['Age_NaN'][i] = 1
df_train.isnull().sum()

#%% カテゴリ変数をonehotで置換
ohe = OneHotEncoder(sparse=False)
dammies = ohe.fit_transform(df_train[['Sex', 'Embarked']])
dammies = pd.DataFrame(dammies)
dammies.columns = np.append(ohe.categories_[0], ohe.categories_[1])

#%% ダミー変数を結合する
after_train = df_train.merge(dammies, right_index=True, left_index=True).drop(
    ['Sex', 'Embarked'], axis=1)
# 多重共線性を考慮してダミー変数を1つずつ消しておく
after_train = after_train.drop(['female', 'C'], axis=1)

after_train.head()


#%% テストデータにも同様の処理を行う
df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)
age_nan = df_test[df_test['Age'].isnull()]
df_test['Age'] = df_test['Age'].fillna(df_test['Age'].mean())
df_test['Age_NaN'] = 0
for i in age_nan.index:
    df_test['Age_NaN'][i] = 1
df_test['Embarked'] = df_test['Embarked'].fillna('S')
df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].median())
# 欠損値の確認
df_test.isnull().sum()

# one-hot-encoding
dammies = ohe.transform(df_test[['Sex', 'Embarked']])
dammies = pd.DataFrame(dammies)
dammies.columns = np.append(ohe.categories_[0], ohe.categories_[1])

after_test = df_test.merge(dammies, right_index=True, left_index=True).drop(
    ['Sex', 'Embarked'], axis=1)
# 多重共線性を考慮してダミー変数を1つずつ消しておく
after_test = after_test.drop(['female', 'C'], axis=1)
after_test.head()
