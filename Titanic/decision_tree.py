'''kaggle titanic'''
# どのような人が生き残りやすかったか
%cd /Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Kaggle/Titanic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rcParams['font.family'] = 'Hiragino Sans'
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score


'''データ前処理'''
#%% データの読み込み & 前処理
df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')
df_ans = pd.read_csv('./data/gender_submission.csv')

# 大雑把な概要を把握
df_train.describe()
df_test.describe()

# 欠損値の確認
df_train.isnull().sum()
df_test.isnull().sum()

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
# 1. 決定木モデル
# ここでは、Pclass, Sex, Age, Fareの4つの項目を使う

#%% trainの目的変数と説明変数の値を取得
target = df_train['Survived'].values
features_one = df_train[['Pclass', 'Sex', 'Age', 'Fare']].values

#%% 決定木の作成
tree_model = DecisionTreeClassifier()
tree_model.fit(features_one, target)
plot_tree(tree_model)
plt.show()

train_pred = tree_model.predict(features_one)
print('trainデータでの正解率:', accuracy_score(target, train_pred))

# テストデータでの予測
test_target = df_ans['Survived'].values
test_features = df_test[['Pclass', 'Sex', 'Age', 'Fare']].values

test_pred = tree_model.predict(test_features)
print('testデータでの正解率:', accuracy_score(test_target, test_pred))


#%% 予測データをcsvで保存してkaggleに提出
PassengerId = df_test['PassengerId'].values

ans_df = pd.DataFrame(test_pred, PassengerId, columns=['Survived'])
ans_df.index.name = 'PassengerId'

ans_df.to_csv('titanic_answer.csv')


'''予測モデルの構築2'''
# 2. 決定木モデル
# 7つの変数に拡張してみる

#%% trainの目的変数と説明変数の値を取得
features_one = df_train[
        ["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values

#%% 決定木の作成
tree_model2 = DecisionTreeClassifier(max_depth=10, min_samples_split=5)
tree_model2.fit(features_one, target)

train_pred2 = tree_model2.predict(features_one)

print('trainデータでの正解率:', accuracy_score(target, train_pred2))

# テストデータでの予測
test_features2 = df_test[
        ["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values

test_pred2 = tree_model2.predict(test_features2)
print('testデータでの正解率:', accuracy_score(test_target, test_pred2))


#%% 回答をcsvに変換
ans_df2 = pd.DataFrame(test_pred2, PassengerId)
ans_df2.columns = ['Survived']
ans_df2.index.name = 'PassengerId'

ans_df2.to_csv('titanic_answer.csv')
