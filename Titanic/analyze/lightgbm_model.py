'''kaggle titanic'''
# どのような人が生き残りやすかったか
import os, sys
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Kaggle/Titanic/analyze')
sys.path.append('../../..')
import numpy as np
np.set_printoptions(precision=3)

from sklearn.model_selection import train_test_split, cross_val_score
import lightgbm as lgb
from sklearn.metrics import confusion_matrix

from function import *
from Function.preprocess_func import *


'''データ前処理'''
#%% データの読み込み & 前処理
df_train = pd.read_csv('../data/train.csv')
df_test = pd.read_csv('../data/test.csv')
drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin']
category_features = ['Sex', 'Embarked']

X_train, y_train, X_test = preprocessing(df_train, df_test,
                                        drop_columns, category_features)

#%% ラベルエンコーディング
X_train, X_test = encoding('label', X_train, X_test, category_features)
# モデルの分割
X_learn, X_val, y_learn, y_val = train_test_split(X_train, y_train,
                                    shuffle=True, stratify=y_train)


'''lightgbmモデルの構築'''
#%% 前準備
clf = lgb.LGBMClassifier() #デフォルト：objective='binary', metrics='binary_logloss'
cross_val_score(clf, X_train, y_train, cv=5) # 学習のばらつきの確認

# 学習と予測
clf.fit(X_learn, y_learn, eval_set=[(X_val, y_val)],
        early_stopping_rounds=20, verbose=10)

val_pred = clf.predict(X_val, num_iteration=clf.best_iteration_)
confusion_matrix(y_val, val_pred)

#%% 変数重要度
lgb_importance(X_learn, clf)


'''テストデータで予測, Submit'''
#%%
ans = test_predict(clf, X_test)
ans.head()
# ファイル保存
ans.to_csv('./data/answer.csv', index=False)


#%% ==============================================================================
'''pseudo labeling'''
pseudo_label = pd.Series(test_pred.round(), name='Survived')

X_train_mix = pd.concat([X_train, X_test], ignore_index=True)
y_train_mix = pd.concat([y_train, pseudo_label], ignore_index=True)

X_learn, X_val, y_learn, y_val = train_test_split(X_train_mix, y_train_mix,
                                    shuffle=True, stratify=y_train_mix)


# モデル作成
clf = lgb.LGBMClassifier()
cross_val_score(clf, X_train_mix, y_train_mix, cv=5)

clf.fit(X_learn, y_learn, eval_set=[(X_val, y_val)],
        early_stopping_rounds=10, verbose=10)

val_pred = clf.predict(X_val, num_iteration=clf.best_iteration_)
confusion_matrix(y_val, val_pred)


# テスト予測
test_pred = clf.predict(X_test, num_iteration=clf.best_iteration_)
pred_df = pd.DataFrame(test_pred.round().astype('int8'), columns=['Survived'])
ans = pd.DataFrame(df_test['PassengerId'], columns=['PassengerId'])
ans = ans.merge(pred_df, right_index=True, left_index=True)
ans.head()

ans.to_csv('../data/answer.csv', index=False)
