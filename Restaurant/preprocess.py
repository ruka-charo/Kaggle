'''kaggle Restaurant'''
# 年間のレストラン売り上げの予測
%cd /Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Kaggle/Restaurant
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 50)
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.style.use('seaborn')

from sklearn.preprocessing import OneHotEncoder


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


# trainデータの調査
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

# カウント 4or5 以下はその他でひとまとめにするのが良さそう
for city in train_data['City'].value_counts().index[5:]:
    train_data['City'] = train_data['City'].replace(city, 'Other')

city_rev = train_data.groupby('City')['revenue'].mean()
plt.bar(city_rev.index, city_rev.values)
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
    ['City', 'City Group', 'Type'], axis=1)

after_train.head()

#%% 
