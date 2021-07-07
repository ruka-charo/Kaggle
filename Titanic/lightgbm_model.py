'''kaggle titanic'''
# どのような人が生き残りやすかったか
import os
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Kaggle/Titanic')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rcParams['font.family'] = 'Hiragino Sans'
import seaborn as sns

import category_encoders as ce
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, roc_auc_score


'''データ前処理'''
#%% データの読み込み & 前処理
df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')

#%% データの取捨選択
train_data = df_train.copy().drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
train_data.head()
train_data.shape

#%% 欠損値の確認
train_data.isnull().sum()

#%% 年齢を平均値で埋める
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())

#%% 目的変数と説明変数に分ける
X_train = train_data.copy().drop(['Survived'], axis=1)
y_train = train_data['Survived']

#%% カテゴリ変数をcategory_labelingで置換
# 今回はlightgbmでの予測なのでlabelencoderで問題ない
category_features = ['Sex', 'Embarked']
oe = ce.OrdinalEncoder(cols=category_features, handle_unknown='impute')
X_train = oe.fit_transform(X_train)
oe.category_mapping
X_train.head()

#%% adversarial validationの準備としてtrainかtestかを表す新たな特徴量を作成する
X_train['Test'] = 0


'''testデータの前処理'''
#%% テストデータにも同様の処理を行う
test_data = df_test.copy().drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
#%% 年齢を平均値で、Fareを中央値で埋める
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())
#%% カテゴリ変数を置換する
test_data = oe.transform(test_data)
test_data['Test'] = 1
test_data.head()


'''adversarial validation'''
# 手元のスコアとLBのスコアに乖離があったためこの手法を試す
#%% trainデータとtestデータが同じ分布に従っているか確認する
mix_data = pd.concat([X_train, test_data], ignore_index=True)
mix_data.shape
mix_data.head()

X_mix = mix_data.copy().drop(['Test'], axis=1)
y_mix = mix_data['Test']

X_mix_learn, X_mix_val, y_mix_learn, y_mix_val = train_test_split(
        X_mix, y_mix, test_size=0.2, shuffle=True, stratify=y_mix)

# lightgbmモデルの構築
#%% 専用のデータ型に変更する
lgb_train = lgb.Dataset(X_mix_learn, label=y_mix_learn, free_raw_data=False)
lgb_val = lgb.Dataset(X_mix_val, label=y_mix_val, free_raw_data=False)

# ハイパーパラメータの設定
params = {'objective': 'binary', 'verbose': 0, 'metrics': 'binary_logloss'}

#%% 学習の実行
model = lgb.train(params, lgb_train, 100,
                valid_sets=[lgb_train, lgb_val],
                valid_names=['train', 'valid'],
                categorical_feature=category_features,
                early_stopping_rounds=10,
                verbose_eval=10)

# バリデーションデータでのスコアを確認
val_pred = model.predict(X_mix_val, num_iteration=model.best_iteration)
print('AUC値:', roc_auc_score(y_mix_val, val_pred))
print('log_loss値:', log_loss(y_mix_val, val_pred))
print('accuracy_score:', accuracy_score(y_mix_val, val_pred.round()))
print('confusion_matrix\n', confusion_matrix(y_mix_val, val_pred.round()))

#%% 変数重要度
# split: 頻度、gain: 目的関数の現象寄与率
importance_type = 'gain'
features = X_mix_learn.columns
importances = model.feature_importance(importance_type=importance_type)
indices = np.argsort(-importances)
plt.bar(np.array(features)[indices], np.array(importances[indices]))
plt.xticks(rotation=90)
plt.title(importance_type)
plt.show()

#%% trainデータとtestデータの分類に年齢が大きく関わっている
plt.hist(X_train['Age'], bins=10, density=True)
plt.hist(test_data['Age'], bins=10, density=True, alpha=0.5)
plt.show()


'''lightgbmモデルの構築'''
#%% 前準備
X_learn, X_val, y_learn, y_val = train_test_split(X_train.drop(['Test'], axis=1),
                                        y_train, shuffle=True, stratify=y_train)

lgb_train = lgb.Dataset(X_learn, label=y_learn, free_raw_data=False)
lgb_val = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

# ハイパーパラメータの設定
params = {'objective': 'binary', 'verbose': 0, 'metrics': 'binary_logloss'}

#%% モデルの実装
model = lgb.train(params, lgb_train, 100,
                valid_sets=[lgb_train, lgb_val],
                valid_names=['train', 'valid'],
                categorical_feature=category_features,
                early_stopping_rounds=10,
                verbose_eval=10)
val_pred = model.predict(X_val, num_iteration=model.best_iteration)


print('AUC値:', roc_auc_score(y_val, val_pred))
print('log_loss値:', log_loss(y_val, val_pred))
print('accuracy_score:', accuracy_score(y_val, val_pred.round()))
print('confusion_matrix\n', confusion_matrix(y_val, val_pred.round()))

#%% 変数重要度
features = X_learn.columns
importance = model.feature_importance('gain')
indices = np.argsort(-importances)
plt.bar(np.array(features)[indices], np.array(importances[indices]))
plt.xticks(rotation=90)
plt.title('gain')
plt.show()


'''テストデータで予測, Submit'''
#%%
test_data = test_data.drop(['Test'], axis=1)

test_pred = model.predict(test_data, num_iteration=model.best_iteration)
pred_df = pd.DataFrame(test_pred.round().astype('int8'), columns=['Survived'])
ans = pd.DataFrame(df_test['PassengerId'], columns=['PassengerId'])
ans = ans.merge(pred_df, right_index=True, left_index=True)
ans.head()

ans.to_csv('./data/answer.csv', index=False)
