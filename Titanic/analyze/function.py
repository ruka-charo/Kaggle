'''データ前処理'''
import pandas as pd
pd.set_option('display.max_columns', 500)
import category_encoders as ce


# 欠損値やカラムの取捨選択
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

    return X_train, y_train, X_test


# テストデータの予測
def test_predict(clf, X_test):
    test_pred = clf.predict(X_test, num_iteration=clf.best_iteration_)
    pred_df = pd.DataFrame(test_pred.round().astype('int8'), columns=['Survived'])
    ans = pd.DataFrame(df_test['PassengerId'], columns=['PassengerId'])
    ans = ans.merge(pred_df, right_index=True, left_index=True)

    return ans
