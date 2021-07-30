'''kaggle titanic'''
# どのような人が生き残りやすかったか
import os, sys
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Kaggle/Titanic')
sys.path.append('../..')
import numpy as np
np.set_printoptions(precision=3)
import pandas as pd
pd.set_option('display.max_columns', 50)
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rcParams['font.family'] = 'Hiragino Sans'
import seaborn as sns

import category_encoders as ce
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix, \
log_loss, roc_auc_score, roc_curve, recall_score

from Function.preprocess_func import *


'''データ前処理'''
#%% データの読み込み & 前処理
df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')

#%% データの取捨選択
train_data = df_train.copy().drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

#%% 年齢を平均値で埋める
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())

#%% 目的変数と説明変数に分ける
X_train = train_data.copy().drop(['Survived'], axis=1)
y_train = train_data['Survived']
X_train.head()

X_learn, X_val, y_learn, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                shuffle=True, stratify=y_train, random_state=1)

#%% target encodingでカテゴリ変数を変換する
cat_cols = ['Sex', 'Embarked', 'Pclass']
X_learn, X_val = target_encoding(cat_cols, X_learn, y_learn, X_val)
X_learn.head()


'''lightgbmモデルの構築'''
#%% モデルの実装
clf = lgb.LGBMClassifier(objective='binary', metrics='binary_logloss')
clf.fit(X_learn, y_learn, eval_set=[(X_val, y_val)],
        early_stopping_rounds=10, verbose=10)

clf.score(X_val, y_val)

#%% 変数重要度
features = X_learn.columns
importances = clf.feature_importances_
indices = np.argsort(-importances)
plt.bar(np.array(features)[indices], np.array(importances[indices]))
plt.xticks(rotation=90)
plt.show()


'''testデータの予測'''
#%% テストデータにも同様の処理を行う
test_data = df_test.copy().drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
# 年齢を平均値で、Fareを中央値で埋める
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())

X_train, X_test = target_encoding(cat_cols, X_train, y_train, test_data)

clf = lgb.LGBMClassifier(objective='binary', metrics='binary_logloss')
clf.fit(X_train, y_train, verbose=10)

y_pred = clf.predict(X_test, num_iteration=clf.best_iteration_)


'''テストデータで予測, Submit'''
#%%
pred_df = pd.DataFrame(y_pred.round(), columns=['Survived'])
ans = pd.DataFrame(df_test['PassengerId'], columns=['PassengerId'])
ans = ans.merge(pred_df, right_index=True, left_index=True)
ans.head()

ans.to_csv('./data/answer.csv', index=False)
