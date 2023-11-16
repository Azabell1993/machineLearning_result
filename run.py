import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
url = 'https://github.com/Azabell1993/machineLearning/raw/main/bike_rentals.csv'
df_bike = pd.read_csv(url)

#### 1. 데이터 파악하기 ####
# 전체 데이터를 읽어서 출력하고 결과를 'result1.txt'에 저장
with open('result1.txt', 'w') as f:
    print(df_bike, file=f)

#### 2. 데이터 타입 파악 ####
# 데이터 타입을 int, float, object로 표기하여 'result2.txt'에 저장
data_types = df_bike.dtypes
with open('result2.txt', 'w') as f:
    for col, dtype in data_types.items():
        print(f"{col}: {dtype}", file=f)

#### 3. 결측치 확인 ####
# NaN 값을 추가하여 결측치를 만듭니다.
df_bike.loc[df_bike.sample(frac=0.1).index, 'Temperature'] = np.nan
df_bike.loc[df_bike.sample(frac=0.1).index, 'Humidity'] = np.nan

# 결측치를 확인하고 가장 많은 순으로 나열한 후 처음 15개를 출력하여 'result3.txt'에 저장
missing_values = df_bike.isna().sum().sort_values(ascending=False)
with open('result3.txt', 'w') as f:
    print(missing_values, file=f)

#### 4. 결측치 상위 15개 출력 ####
# 결측치가 가장 많은 상위 15개를 출력하여 'result4.txt'에 저장
with open('result4.txt', 'w') as f:
    print(missing_values.head(15), file=f)

#### 5. 카테고리 자료형은 one-hot encoding으로 처리 ####
df_bike = pd.get_dummies(df_bike)

#### 6. 특정 자료형 관련 속성 시각화 및 결과 출력 ####
# SalePrice와의 상관관계가 높은 상위 4개 속성을 시각화하고 'result5.txt'에 저장
df_corr = df_bike.corr()
df_corr_sort = df_corr.sort_values('BikeRentals', ascending=False)
top_correlated_features = df_corr_sort.head(5).index.tolist()

plt.figure(figsize=(10, 8))
sns.pairplot(data=df_bike, vars=top_correlated_features, hue='BikeRentals', palette='husl')
plt.savefig('correlation_plot.png')
plt.show()

# Close the first figure explicitly
plt.close()

# 머신러닝 결과 출력
X_train, X_test, y_train, y_test = train_test_split(
    df_bike.drop('BikeRentals', axis=1),
    df_bike['BikeRentals'].values,
    test_size=0.2,
    random_state=42
)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_imputed, y_train)

# Test
predictions = model.predict(X_test_imputed)

# 결과로 나오는 머신러닝 모습을 'result5.txt'에 저장
with open('result5.txt', 'w') as f:
    print(f'Machine Learning Results:\n\n', file=f)
    print(f'Predictions: {predictions}\n', file=f)
    print(f'True Values: {y_test}\n', file=f)

# 결과를 로그로도 보여주기
print('Machine Learning Results:\n')
print(f'Predictions: {predictions}\n')
print(f'True Values: {y_test}\n')

