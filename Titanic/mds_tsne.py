'''kaggle titanic'''
# どのような人が生き残りやすかったか
%cd /Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Kaggle/Titanic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rcParams['font.family'] = 'Hiragino Sans'
import seaborn as sns

from sklearn.manifold import MDS, TSNE
from sklearn.model_selection import \
    train_test_split, cross_validate, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


'''データ前処理'''
#%% データの読み込み & 前処理
df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')
df_ans = pd.read_csv('./data/gender_submission.csv')

#%% 欠損データを代理データに入れ替える
# Age, Embarked, Cabin
# 今回は Age: 中央値, Embark: 最頻値　で置き換えることにする
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())
df_train['Embarked'] = df_train['Embarked'].fillna('S')
# 欠損値の確認
df_train.isnull().sum()

#%% 文字列データを数字に変換
# 今回は Sex と Embarked を数値化する
df_train['Sex'] = df_train['Sex'].replace({'male': 0, 'female': 1})
df_train['Embarked'] = df_train['Embarked'].replace({'S': 0,
                                                    'C': 1,
                                                    'Q': 2})
# 変換後の確認
df_train.head()

#%% テストデータにも同様の処理を行う
df_test['Age'] = df_test['Age'].fillna(df_test['Age'].median())
df_test['Embarked'] = df_test['Embarked'].fillna('S')
df_test['Sex'] = df_test['Sex'].replace({'male': 0, 'female': 1})
df_test['Embarked'] = df_test['Embarked'].replace({'S': 0,
                                                    'C': 1,
                                                    'Q': 2})
# Fareに欠損値が1つあるので中央値で置き換える
df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].median())
# 欠損値の確認
df_test.isnull().sum()


'''多次元尺度構成法'''
#%% 訓練データの圧縮
features = ["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]
train_features = df_train[features].values
train_target = df_train['Survived']

# 男女合同で
#%% 次元を2に圧縮してみる
mds = MDS(n_components=2)
Y = mds.fit_transform(train_features)

#%% 可視化
plt.scatter(Y[:, 0], Y[:, 1], c=train_target, cmap='bwr')
plt.show()

# 男女別で
#%% 男性
train_male = df_train[features].query('Sex == 0').drop('Sex', axis=1)
target_male = df_train['Survived'][train_male.index]

Y_male = mds.fit_transform(train_male)

# 可視化
plt.scatter(Y_male[:, 0], Y_male[:, 1], c=target_male, cmap='bwr')
plt.show()

#%% 女性
train_female = df_train[features].query('Sex == 1').drop('Sex', axis=1)
target_female = df_train['Survived'][train_female.index]

Y_female = mds.fit_transform(train_female)

# 可視化
plt.scatter(Y_female[:, 0], Y_female[:, 1], c=target_female, cmap='bwr')
plt.show()


'''t-SNE'''
#%% 次元を2に圧縮してみる
tsne = TSNE(n_components=2,
            perplexity=50,
            early_exaggeration=12,
            learning_rate=400,
            method='barnes_hut')
Y = tsne.fit_transform(train_features)

#%% 可視化
plt.scatter(Y[:, 0], Y[:, 1], c=train_target, cmap='bwr')
plt.show()

# 男女別で
#%% 男性
Y_male = tsne.fit_transform(train_male)

# 可視化
plt.scatter(Y_male[:, 0], Y_male[:, 1], c=target_male, cmap='bwr')
plt.show()

#%% 女性
Y_female = tsne.fit_transform(train_female)

# 可視化
plt.scatter(Y_female[:, 0], Y_female[:, 1], c=target_female, cmap='bwr')
plt.show()
