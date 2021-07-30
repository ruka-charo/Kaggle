'''kaggle titanic'''
# どのような人が生き残りやすかったか
import os, sys
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Kaggle/Titanic/analyze')
sys.path.append('../../..')
import numpy as np
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rcParams['font.family'] = 'Hiragino Sans'
import seaborn as sns

from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, roc_auc_score

from function import *
from Function.preprocess_func import *


'''データ前処理'''
#%% データの読み込み & 前処理
df_train = pd.read_csv('../data/train.csv')
df_test = pd.read_csv('../data/test.csv')

drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin']
category_features = ['Sex', 'Embarked']

X_train, X_test = preprocessing(df_train, df_test, drop_columns, category_features)



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
