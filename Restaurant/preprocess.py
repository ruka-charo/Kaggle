'''kaggle Restaurant'''
# 年間のレストラン売り上げの予測
%cd /Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Kaggle/Restaurant
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 50)
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.style.use('seaborn')

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


'''
Id:         レストランのID
Open Date:  レストランのオープン日
City:       レストランが存在する都市 (unicodeに注意)
City Group: 都市グループ
            大都市 or その他
Type:       レストランのタイプ
            FC: フードコート, IL: インライン, DT: ドライブスルー, MB: モバイル
Pi:         人工統計データ, 不動産データ, 商業データ
Revenue:    収益(変換後)
'''


#%% データの読み込み
train_data = pd.read_csv('./data/train.csv.zip')
train_data.head()

test_data = pd.read_csv('./data/test.csv.zip')
test_data.head()


'''trainデータの調査'''
#%% 全体
train_data.describe()
train_data.isnull().any() # 欠損値なし

# 個別に観察
#%% ユニークなCity
city_name = train_data['City'].unique().sort()
train_data['City'].value_counts()

city_rev = train_data.groupby('City')['revenue'].mean()
plt.bar(city_rev.index, city_rev.values)
plt.xticks(rotation=90)
plt.show()

#%% City Group
train_data['City Group'].value_counts()
group_rev = train_data.groupby('City Group')['revenue'].mean()
plt.bar(group_rev.index, group_rev.values)
plt.show()

#%% Type
train_data['Type'].value_counts()
type_rev = train_data.groupby('Type')['revenue'].mean()
plt.bar(type_rev.index, type_rev.values)
plt.show()


'''データ前処理'''
#%% カウント 4or5 以下はその他でひとまとめにするのが良さそう
for city in train_data['City'].value_counts().index[5:]:
    train_data['City'] = train_data['City'].replace(city, 'Other')
    test_data['City'] = test_data['City'].replace(city, 'Other')

city_rev = train_data.groupby('City')['revenue'].mean()
plt.bar(city_rev.index, city_rev.values)
plt.show()

#%% City, CityGroup, TypeをOnehot化する。
ohe = OneHotEncoder(sparse=False)
dammies = ohe.fit_transform(train_data[['City', 'City Group', 'Type']])
dammies = pd.DataFrame(dammies)
columns = np.append(ohe.categories_[0], ohe.categories_[1])
columns = np.append(columns, ohe.categories_[2])
dammies.columns = columns
# 多重共線性を考慮してダミー変数を1つずつ消しておく
dammies = dammies.drop(['Other', 'IL'], axis=1)

#%% ダミー変数を結合する
after_train = train_data.merge(dammies, right_index=True, left_index=True).drop(
    ['Id', 'City', 'City Group', 'Type'], axis=1)

#%% OpenDateについて、年数のみのデータにする
# データ型を日付型に
after_train['Open Date'] = pd.to_datetime(after_train['Open Date'])
# 年数のみ抽出
after_train['Open Year'] = after_train['Open Date'].dt.year

# 年数ごとの利益を可視化
after_train['Open Year'].value_counts()
year_rev = after_train[['Open Year', 'revenue']].groupby('Open Year').mean()

plt.bar(year_rev.index, year_rev['revenue'])
plt.xlabel('西暦')
plt.xticks(rotation=90)
plt.ylabel('利益')
plt.show()

#%% 目的変数と説明変数に分ける
X_train = after_train.copy().drop(['Open Date', 'revenue'], axis=1)
y_train = after_train.copy()['revenue']


'''主成分分析'''
# 標準化
ss = StandardScaler()
X_train_ss = ss.fit_transform(X_train)

# 主成分分析
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X_train_ss)
plt.scatter(np.arange(1, X_train_ss.shape[1]+1),
        np.cumsum(pca.explained_variance_ratio_))
plt.plot(np.arange(1, X_train_ss.shape[1]+1),
        np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('主成分要素数')
plt.ylabel('累積確率')
plt.show()

#%% 20次元までで9.5割再現できる
pca = PCA(n_components=20)
X_pca = pca.fit_transform(X_train_ss)

'''モデル構築'''
#%% 線形回帰
lr = LinearRegression()

lr.fit(X_train_ss, y_train)
y_pred = lr.predict(X_train_ss)

print('r2-score:', r2_score(y_train, y_pred))


#%% ランダムフォレスト
rfr = RandomForestRegressor(n_estimators=500,
                            max_depth=5,
                            max_leaf_nodes=5)
rfr.fit(X_pca, y_train)
y_pred = rfr.predict(X_pca)

print('r2-score:', r2_score(y_train, y_pred))

#%% 変数重要度
importances = rfr.feature_importances_
indices = np.argsort(-importances)
plt.bar(np.array(X_train.columns)[indices], np.array(rfr.feature_importances_[indices]))
plt.xticks(rotation=90)
plt.show()


# テストにしかないカテゴリがあって、前処理は一工夫しなければいけない
