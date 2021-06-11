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
from sklearn.preprocessing import OneHotEncoder
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

#%% 文字列データを数字に変換 (one_hot_encoder)
# 今回は Sex と Embarked を数値化する
'''多重共線性の影響を見たいので次元はそのまま'''
ohe = OneHotEncoder(sparse=False)
dammies = ohe.fit_transform(df_train[['Sex', 'Embarked']])

df_dammies = pd.DataFrame(dammies,
    columns=np.append(ohe.categories_[0], ohe.categories_[1]))
df_train = df_train.drop(columns=['Sex', 'Embarked']).merge(df_dammies,
                                                            left_index=True,
                                                            right_index=True)
# 変換後の確認
df_train.head()

#%% テストデータにも同様の処理を行う
df_test['Age'] = df_test['Age'].fillna(df_test['Age'].median())
df_test['Embarked'] = df_test['Embarked'].fillna('S')

dammies_test = ohe.transform(df_test[['Sex', 'Embarked']])
df_dammies_test = pd.DataFrame(dammies_test,
    columns=np.append(ohe.categories_[0], ohe.categories_[1]))
df_test = df_test.drop(columns=['Sex', 'Embarked']).merge(df_dammies_test,
                                                            left_index=True,
                                                            right_index=True)
# 変換後の確認
df_test.head()

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

features = ["Pclass", "Age", "Fare", "SibSp",
            "Parch", 'female', 'male', 'C', 'Q', 'S']
train_features = df_train[features].values
train_target = df_train['Survived']

rfc.fit(train_features, train_target)

train_pred = rfc.predict(train_features)
print('trainデータでの正解率:', accuracy_score(train_target, train_pred))
print(confusion_matrix(train_target, train_pred))
print('F1-score:', f1_score(train_target, train_pred))


#%% テストデータでの予測
test_features = df_test[features].values
test_target = df_ans['Survived']

test_pred = rfc.predict(test_features)
print('testデータでの正解率:', accuracy_score(test_target, test_pred))
print(confusion_matrix(test_target, test_pred))
print('F1-score:', f1_score(test_target, test_pred))


#%% 変数重要度
importances = rfc.feature_importances_
indices = np.argsort(-importances)
plt.bar(np.array(features)[indices], np.array(rfc.feature_importances_[indices]))
plt.savefig('random_onehot.png')
plt.show()


#%% 次元を減らした場合
'''多重共線性を考慮し、次元を1つ削除'''
# 訓練データ
df_train_d = df_train.drop(['female', 'C'], axis=1)
# 変換後の確認
df_train_d.head()

# テストデータにも同様の処理を行う
df_test_d = df_test.drop(['female', 'C'], axis=1)
# 変換後の確認
df_test_d.head()
# Fareに欠損値が1つあるので中央値で置き換える
df_test_d['Fare'] = df_test_d['Fare'].fillna(df_test_d['Fare'].median())

#%% 訓練データでの学習と精度
rfc = RandomForestClassifier(n_estimators=500,
                            min_samples_split=25,
                            min_samples_leaf=2,
                            max_leaf_nodes=7,
                            max_depth=4)

features_d = ["Pclass", "Age", "Fare", "SibSp",
            "Parch", 'male', 'Q', 'S']
train_d_features = df_train_d[features_d].values
train_d_target = df_train_d['Survived']

rfc.fit(train_d_features, train_d_target)

train_d_pred = rfc.predict(train_d_features)
print('次元を減らした時のtrain正解率:', accuracy_score(train_d_target, train_d_pred))
print(confusion_matrix(train_d_target, train_d_pred))
print('F1-score:', f1_score(train_d_target, train_d_pred))

#%% テストデータで予測
test_d_features = df_test_d[features_d].values
test_d_target = df_ans['Survived']

test_d_pred = rfc.predict(test_d_features)
print('次元を減らした時のtest正解率:', accuracy_score(test_d_target, test_d_pred))
print(confusion_matrix(test_d_target, test_d_pred))
print('F1-score:', f1_score(test_d_target, test_d_pred))

#%% 変数重要度
importances = rfc.feature_importances_
indices = np.argsort(-importances)
plt.bar(np.array(features_d)[indices], np.array(rfc.feature_importances_[indices]))
plt.savefig('random_onehot_delete.png')
plt.show()


#%% csvにして書き出す
PassengerId = df_test['PassengerId'].values

ans_df = pd.DataFrame(test_d_pred, PassengerId)
ans_df.columns = ['Survived']
ans_df.index.name = 'PassengerId'

ans_df.to_csv('titanic_answer.csv')
