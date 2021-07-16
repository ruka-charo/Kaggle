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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, roc_auc_score



'''データ前処理'''
#%% データの読み込み & 前処理
df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')

# データの取捨選択
train_data = df_train.copy().drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# 年齢を平均値で埋める
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())

# 目的変数と説明変数に分ける
X_train = train_data.copy().drop(['Survived'], axis=1)
y_train = train_data['Survived']

# カテゴリ変数をcategory_labelingで置換
category_features = ['Sex', 'Embarked']
oe = ce.OneHotEncoder(cols=category_features, handle_unknown='impute')
X_train = oe.fit_transform(X_train).drop(['Sex_1', 'Embarked_4'], axis=1)
oe.category_mapping
X_train.head()
X_train.columns
category_features = ['Sex_2', 'Embarked_1', 'Embarked_2', 'Embarked_3']


'''testデータの前処理'''
#%% テストデータにも同様の処理を行う
test_data = df_test.copy().drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
# 年齢を平均値で、Fareを中央値で埋める
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())
# カテゴリ変数を置換する
test_data = oe.transform(test_data).drop(['Sex_1', 'Embarked_4'], axis=1)
test_data.head()


#%% データを分割する
X_learn, X_val, y_learn, y_val = train_test_split(
        X_train, y_train, test_size=0.2, shuffle=True, stratify=y_train)


'''lightgbmモデル'''
# lightgbmモデルの構築
#%% 専用のデータ型に変更する
lgb_train = lgb.Dataset(X_learn, label=y_learn, free_raw_data=False)
lgb_val = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

# ハイパーパラメータの設定
params1 = {'objective': 'binary', 'verbose': 0,
        'metrics': 'binary_logloss', 'max_depth': 4}

# 学習の実行
model1 = lgb.train(params1, lgb_train, 100,
                valid_sets=[lgb_train, lgb_val],
                valid_names=['train', 'valid'],
                categorical_feature=category_features,
                early_stopping_rounds=10,
                verbose_eval=10)

# バリデーションデータでのスコアを確認
light_val_pred1 = model1.predict(X_val, num_iteration=model1.best_iteration)
light_val_pred1

#%%
# ハイパーパラメータの設定
params2 = {'objective': 'binary', 'verbose': 0,
        'metrics': 'binary_logloss', 'max_depth': 8}

# 学習の実行
model2 = lgb.train(params2, lgb_train, 100,
                valid_sets=[lgb_train, lgb_val],
                valid_names=['train', 'valid'],
                categorical_feature=category_features,
                early_stopping_rounds=10,
                verbose_eval=10)

# バリデーションデータでのスコアを確認
light_val_pred2 = model2.predict(X_val, num_iteration=model2.best_iteration)
light_val_pred2

#%% ハイパーパラメータの設定
params3 = {'objective': 'binary', 'verbose': 0,
        'metrics': 'binary_logloss'}

# 学習の実行
model3 = lgb.train(params3, lgb_train, 100,
                valid_sets=[lgb_train, lgb_val],
                valid_names=['train', 'valid'],
                categorical_feature=category_features,
                early_stopping_rounds=10,
                verbose_eval=10)

# バリデーションデータでのスコアを確認
light_val_pred3 = model3.predict(X_val, num_iteration=model3.best_iteration)
light_val_pred3

'''ニューラルネットワーク'''
#%% モデルの構築
def create_model(optimizer='adam', init='glorot_normal'):
    # 下のようにLayerをaddで積み重ねるようにしてNNの全体を構成する
    model = Sequential()
    # inputの次元は明示的に120と書くよりもX.shape[1]と書くのが一般的
    model.add(Dense(16, input_dim=X_learn.shape[1], kernel_initializer=init, activation='relu'))
    model.add(Dense(8, kernel_initializer=init, activation='relu'))
    model.add(Dense(4, kernel_initializer=init, activation='relu'))
    # 最後のNNの出力は1次元のスカラーが出力される。それにsigmoid関数をかける
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))

    # 2値分類なのでbinary_crossentropyを使う
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

model4 = create_model()

# コールバック設定
# 3回変化がなかったら止める
es = EarlyStopping(monitor='val_loss', patience=10,
                    verbose=1, restore_best_weights=True)

# epoch数にこだわる必要なし
model4.fit(x=X_learn, y=y_learn,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=100,
        callbacks=[es],
        shuffle=True)

nn_val_pred = model4.predict(X_val).reshape(-1)
nn_val_pred


#%% モデルの構築
def create_model(optimizer='adam', init='glorot_normal'):
    # 下のようにLayerをaddで積み重ねるようにしてNNの全体を構成する
    model = Sequential()
    # inputの次元は明示的に120と書くよりもX.shape[1]と書くのが一般的
    model.add(Dense(64, input_dim=X_learn.shape[1], kernel_initializer=init, activation='relu'))
    model.add(Dense(32, kernel_initializer=init, activation='relu'))
    model.add(Dense(16, kernel_initializer=init, activation='relu'))
    model.add(Dense(8, kernel_initializer=init, activation='relu'))
    model.add(Dense(4, kernel_initializer=init, activation='relu'))
    # 最後のNNの出力は1次元のスカラーが出力される。それにsigmoid関数をかける
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))

    # 2値分類なのでbinary_crossentropyを使う
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

model5 = create_model()

# コールバック設定
# 3回変化がなかったら止める
es = EarlyStopping(monitor='val_loss', patience=10,
                    verbose=1, restore_best_weights=True)

# epoch数にこだわる必要なし
model5.fit(x=X_learn, y=y_learn,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=100,
        callbacks=[es],
        shuffle=True)

nn_val_pred2 = model5.predict(X_val).reshape(-1)
nn_val_pred2


#%%
pred_dict = {'lightgbm1': light_val_pred1,
            'lightgbm2': light_val_pred2,
            'lightgbm3': light_val_pred3,
            'NN1': nn_val_pred,
            'NN2': nn_val_pred2}


pred_proba = (pred_dict['lightgbm1'] + pred_dict['lightgbm2'] +
            pred_dict['lightgbm3'] + pred_dict['NN1'] +
            pred_dict['NN2']) / len(pred_dict)

pred_val = pd.DataFrame(pred_proba.round(), columns=['Survived'])
print(f'accuracy: {accuracy_score(pred_val, y_val)}')
confusion_matrix(pred_val, y_val)




'''テストデータで予測, Submit'''
#%%
test_pred1 = model1.predict(test_data, num_iteration=model1.best_iteration)
test_pred2 = model2.predict(test_data, num_iteration=model2.best_iteration)
test_pred3 = model3.predict(test_data, num_iteration=model3.best_iteration)
test_pred4 = model4.predict(test_data).reshape(-1)
test_pred5 = model5.predict(test_data).reshape(-1)

pred_proba = (test_pred1 + test_pred2 + test_pred3 + test_pred4 + test_pred5) / 5

pred_df = pd.DataFrame(pred_proba.round().astype('int8'), columns=['Survived'])
ans = pd.DataFrame(df_test['PassengerId'], columns=['PassengerId'])
ans = ans.merge(pred_df, right_index=True, left_index=True)
ans.head()

ans.to_csv('./data/answer.csv', index=False)
