'''kaggle titanic'''
# どのような人が生き残りやすかったか
%cd /Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Kaggle/Titanic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rcParams['font.family'] = 'Hiragino Sans'
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import \
    cross_validate, KFold, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


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


#%% 訓練データの調整
after_train.query('Survived == 0 and male == 1')
after_train.query('Survived == 1 and male == 1')
after_train.query('Survived == 0 and male == 0')
after_train.query('Survived == 1 and male == 0')

#%% 男性死亡者が非常に多く、偏っているためダウンサンプリングを行う
# 今回は女性のデータの割合に合わせる : 468 → 268 (男性死亡者データ)
after_train_1 = after_train.copy().drop(
                        after_train.query('Survived == 0 and male == 1').index)
after_train_2 = after_train.query('Survived == 0 and male == 1').sample(n=268)

after_train_3 = pd.concat([after_train_1, after_train_2],
                            ignore_index=True).sample(frac=1)
after_train_3.head()

#%% 訓練データと検証用データに分ける
features = ['Pclass', 'Age', 'SibSp', 'Parch',
            'Fare', 'Age_NaN', 'male', 'Q', 'S']

X_train, X_val, y_train, y_val = train_test_split(after_train_3[features],
                                                after_train_3['Survived'],
                                                test_size=0.2)

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

clf.fit(X_train, y_train)
print('ベストパラメータ:', clf.best_params_)
print('ベストスコア:', clf.best_score_)

#%% ランダムフォレストでやってみる
# モデル構築
rfc = RandomForestClassifier(n_estimators=1000,
                            min_samples_split=20,
                            min_samples_leaf=2,
                            max_leaf_nodes=7,
                            max_depth=4)

# 学習
rfc.fit(X_train, y_train)

# 予測
y_pred = rfc.predict(X_val)

# 結果
print('valデータでの正解率:', accuracy_score(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))
print('F1-score:', f1_score(y_val, y_pred))

#%% 変数重要度
importances = rfc.feature_importances_
indices = np.argsort(-importances)
plt.bar(np.array(features)[indices], np.array(rfc.feature_importances_[indices]))
plt.show()



#%% テストデータでの予測
test_pred = rfc.predict(after_test[features])
test_pred = pd.DataFrame(test_pred, columns=['pred'])


#%% csvにして書き出す
ans_df = pd.DataFrame(test_pred['pred'].values, after_test['PassengerId'].values)
ans_df.columns = ['Survived']
ans_df.index.name = 'PassengerId'

ans_df.to_csv('titanic_answer.csv')
