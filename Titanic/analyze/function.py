'''データ前処理'''
import pandas as pd
pd.set_option('display.max_columns', 500)
import category_encoders as ce



def preprocessing(train_data, test_data, drop_columns, category_features):
    # データの取捨選択
    train_data = train_data.drop(drop_columns, axis=1)
    # 年齢を平均値で埋める
    train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())

    # 目的変数と説明変数に分ける
    X_train = train_data.drop(['Survived'], axis=1)
    y_train = train_data['Survived']

    #テストデータにも同様の処理を行う
    X_test = test_data.drop(drop_columns, axis=1)
    # 年齢を平均値で、Fareを中央値で埋める
    X_test['Age'] = X_test['Age'].fillna(X_test['Age'].mean())
    X_test['Fare'] = X_test['Fare'].fillna(X_test['Fare'].median())

    X_train, X_test = _label_encoding(X_train, X_test, category_features)

    return X_train, X_test



def _label_encoding(X_train, X_test, category_features):
    # データを結合させる
    mix_df = pd.concat([X_train, X_test])

    # # カテゴリ変数をcategory_labelingで置換
    category_features = category_features
    oe = ce.OrdinalEncoder(cols=category_features)
    mix_df = oe.fit_transform(mix_df)

    # データを分割する
    X_train_en = mix_df[:X_train.shape[0]]
    X_test_en = mix_df[X_train.shape[0]:]

    return X_train_en, X_test_en
