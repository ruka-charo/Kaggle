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
from sklearn.metrics import accuracy_score, confusion_matrix, \
log_loss, roc_auc_score, roc_curve, recall_score


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

#%% カテゴリ変数をcategory_labelingで置換
# 今回はlightgbmでの予測なのでlabelencoderで問題ない
category_features = ['Sex', 'Embarked']
oe = ce.OrdinalEncoder(cols=category_features, handle_unknown='impute')
X_train = oe.fit_transform(X_train)
oe.category_mapping
X_train.head()
X_train.shape


'''testデータの前処理'''
#%% テストデータにも同様の処理を行う
test_data = df_test.copy().drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
# 年齢を平均値で、Fareを中央値で埋める
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())
# カテゴリ変数を置換する
test_data = oe.transform(test_data)
test_data.head()
test_data.shape


'''lightgbmモデルの構築'''
#%% 前準備
X_learn, X_val, y_learn, y_val = train_test_split(X_train, y_train,
                                shuffle=True, stratify=y_train, random_state=1)

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

#%% ROCカーブで適切なthresholdsを見つける
fpr, tpr, thresholds = roc_curve(y_val, val_pred)

plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR(recall)')
# 0.5に最も近いスレッショルドを見つける
close_zero = np.argmin(np.abs(thresholds - 0.5))
plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10,
        label='threshold 0.5', fillstyle='none', c='k', mew=2)
plt.legend()
plt.show()

#%% recallを0.8くらいまであげてみる
# 生存確率 = 0.3以上ならば生きているとする
for i in np.arange(0.1, 0.9, 0.1):
    func = lambda x: 0 if x <= i else 1
    val_df = pd.DataFrame(val_pred, columns=['val_pred'])
    val_pred_int = val_df.applymap(func)
    print(accuracy_score(y_val, val_pred_int))
recall_score(y_val, val_pred_int)
confusion_matrix(y_val, val_pred_int)



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
test_pred = model.predict(test_data, num_iteration=model.best_iteration)
pred_df = pd.DataFrame(test_pred, columns=['Survived']).applymap(func)
ans = pd.DataFrame(df_test['PassengerId'], columns=['PassengerId'])
ans = ans.merge(pred_df, right_index=True, left_index=True)
ans.head()

ans.to_csv('./data/answer.csv', index=False)
