%cd /Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Kaggle/Titanic
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from keras.callbacks import EarlyStopping, ModelCheckpoint


# 初期設定
file_train = './data/train.csv'
seed = 2019
np.random.seed(seed)

#%% データの読み込み
train_df = pd.read_csv(file_train, index_col='PassengerId')
train_df.head()
train_df.isnull().sum()

#%% データ前処理
def prep_data(df):
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    df[['Age']] = df[['Age']].fillna(df[['Age']].mean())
    df[['Fare']] = df[['Fare']].fillna(df[['Fare']].mean())
    df[['Embarked']] = df[['Embarked']].fillna(df['Embarked'].value_counts().idxmax())
    df['Sex'] = df['Sex'].replace({'female': 1, 'male': 0})

    emberked_one_hot = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = df.drop('Embarked', axis=1)
    df = df.join(emberked_one_hot)

    return df

train_df = prep_data(train_df)
train_df.head()


#%% 目的変数と説明変数を分離する
X = train_df.drop(['Survived'], axis=1)

ss = StandardScaler()
X = ss.fit_transform(X)
y = train_df['Survived']

# テストデータと検証データに分ける
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=seed)

#%% モデルの構築
def create_model(optimizer='adam', init='glorot_normal'):
    # 下のようにLayerをaddで積み重ねるようにしてNNの全体を構成する
    model = Sequential()
    # inputの次元は明示的に120と書くよりもX.shape[1]と書くのが一般的
    model.add(Dense(16, input_dim=X.shape[1], kernel_initializer=init, activation='relu'))
    model.add(Dense(8, kernel_initializer=init, activation='relu'))
    model.add(Dense(4, kernel_initializer=init, activation='relu'))
    # 最後のNNの出力は1次元のスカラーが出力される。それにsigmoid関数をかける
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))

    # 2値分類なのでbinary_crossentropyを使う
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

model = create_model()

# モデルについて
model.summary()
X_train.shape

#%% モデルの学習
model.fit(x=X_train, y=y_train,
        validation_data=(X_test, y_test),
        batch_size=32,
        epochs=20,
        shuffle=True)
pred = model.predict(X_test)
# 予測結果を四捨五入する
pred_round = np.zeros(len(X_test))
for i, num in enumerate(pred[0]):
    pred_round[i] = round(num)

accuracy_score(y_test, pred_round)
roc_auc_score(y_test, pred)


#%% モデル学習にコールバックを使ってみる
model = create_model()

# コールバック設定
# 3回変化がなかったら止める
es = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
# 最良のモデルを保存する
cp = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss',
                    save_best_only=True, verbose=1)

# epoch数にこだわる必要なし
model.fit(x=X_train, y=y_train,
        validation_data=(X_test, y_test),
        batch_size=32,
        epochs=50,
        callbacks=[es, cp],
        shuffle=True)

    
