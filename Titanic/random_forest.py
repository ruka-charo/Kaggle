'''kaggle titanic'''
# どのような人が生き残りやすかったか
%cd /Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Kaggle/Titanic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rcParams['font.family'] = 'Hiragino Sans'
import seaborn as sns

from sklearn.model_selection import \
    cross_validate, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
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


'''予測モデルの構築'''
# ランダムフォレスト
#%% グリッドサーチで良いパラメータを選ぶ
param_grid = {'min_samples_split': [15, 20, 25, 30],}

clf = GridSearchCV(
    estimator=rfc,
    param_grid=param_grid,
    scoring='accuracy',
    cv=KFold(n_splits=5, shuffle=True),
    return_train_score=True)

clf.fit(train_features, train_target)
print('ベストパラメータ:', clf.best_params_)
print('ベストスコア:', clf.best_score_)

#%% ランダムフォレストでやってみる
rfc = RandomForestClassifier(n_estimators=500,
                            min_samples_split=25,
                            min_samples_leaf=2,
                            max_leaf_nodes=7,
                            max_depth=4)

features = ["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]
train_features = df_train[features].values
train_target = df_train['Survived']

rfc.fit(train_features, train_target)

train_pred = rfc.predict(train_features)
print('trainデータでの正解率:', accuracy_score(train_target, train_pred))
confusion_matrix(train_target, train_pred)
print('F1-score:', f1_score(train_target, train_pred))


#%% テストデータでの予測
test_features = df_test[features].values
test_target = df_ans['Survived']

test_pred = rfc.predict(test_features)
print('testデータでの正解率:', accuracy_score(test_target, test_pred))
confusion_matrix(test_target, test_pred)
print('F1-score:', f1_score(test_target, test_pred))

#%% 変数重要度
importances = rfc.feature_importances_
indices = np.argsort(-importances)
plt.bar(np.array(features)[indices], np.array(rfc.feature_importances_[indices]))
plt.show()


#%% csvにして書き出す
PassengerId = df_test['PassengerId'].values

ans_df = pd.DataFrame(test_pred, PassengerId)
ans_df.columns = ['Survived']
ans_df.index.name = 'PassengerId'

ans_df.to_csv('titanic_answer.csv')
