# 필요한 라이브러리를 불러옵니다.
import pandas as pd  # 데이터 조작 및 분석을 위한 라이브러리
import numpy as np  # 수학 및 과학 계산을 위한 라이브러리
from sklearn.model_selection import train_test_split  # 데이터를 훈련 및 테스트 세트로 나누는 데 사용되는 함수
from sklearn.ensemble import RandomForestRegressor  # 랜덤 포레스트 회귀 모델
from sklearn.impute import SimpleImputer  # 결측치를 채우기 위한 라이브러리
import matplotlib.pyplot as plt  # 시각화를 위한 라이브러리
import seaborn as sns  # 시각화를 위한 라이브러리

# 데이터를 인터넷에서 불러와서 DataFrame에 저장합니다.
url = 'https://github.com/Azabell1993/machineLearning/raw/main/bike_rentals.csv'
df_bike = pd.read_csv(url)

#### 1. 데이터 파악하기 ####
# 전체 데이터를 출력하고 결과를 'data_overview.txt'에 저장합니다.
with open('data_overview.txt', 'w') as f:
    print(df_bike, file=f)

# 출력된 데이터를 코랩에서 확인합니다.
print("Data Overview:")

# colab에서 실행 중인지 확인하여 display 함수를 사용할지 print 함수를 사용할지 결정합니다.
try:
    from IPython.display import display
    display(df_bike.head())
except ImportError:
    print(df_bike.head())

#### 2. 데이터 타입 파악 ####
# 데이터의 타입을 확인하고 'data_types.txt'에 저장합니다.
data_types = df_bike.dtypes
with open('data_types.txt', 'w') as f:
    for col, dtype in data_types.items():
        print(f"{col}: {dtype}", file=f)

# 출력된 데이터 타입을 코랩에서 확인합니다.
print("\nData Types:")

# colab에서 실행 중인지 확인하여 display 함수를 사용할지 print 함수를 사용할지 결정합니다.
try:
    from IPython.display import display
    display(data_types)
except ImportError:
    print(data_types)

#### 3. 결측치 확인 ####
# NaN 값을 추가하여 결측치를 만들고 확인한 후 'missing_values.txt'에 저장합니다.
df_bike.loc[df_bike.sample(frac=0.1).index, 'Temperature'] = np.nan
df_bike.loc[df_bike.sample(frac=0.1).index, 'Humidity'] = np.nan
missing_values = df_bike.isna().sum().sort_values(ascending=False)
with open('missing_values.txt', 'w') as f:
    print(missing_values, file=f)

# 결측치 결과를 코랩에서 확인합니다.
print("\nMissing Values:")

# colab에서 실행 중인지 확인하여 display 함수를 사용할지 print 함수를 사용할지 결정합니다.
try:
    from IPython.display import display
    display(missing_values.head(15))
except ImportError:
    print(missing_values.head(15))

#### 4. 결측치 상위 15개 출력 ####
# 결측치가 가장 많은 상위 15개를 출력하고 'top_missing_values.txt'에 저장합니다.
with open('top_missing_values.txt', 'w') as f:
    print(missing_values.head(15), file=f)

# 상위 15개 결측치를 코랩에서 확인합니다.
print("\nTop Missing Values:")

# colab에서 실행 중인지 확인하여 display 함수를 사용할지 print 함수를 사용할지 결정합니다.
try:
    from IPython.display import display
    display(missing_values.head(15))
except ImportError:
    print(missing_values.head(15))

#### 5. 카테고리 자료형은 one-hot encoding으로 처리 ####
# 카테고리형 변수를 더미 변수로 변환합니다.
df_bike = pd.get_dummies(df_bike)


#### 6. 특정 자료형 관련 속성 시각화 및 결과 출력 ####
# SalePrice와의 상관관계가 높은 상위 4개 속성을 시각화하고 'correlation_results.txt'에 저장합니다.
df_corr = df_bike.corr()
df_corr_sort = df_corr.sort_values('BikeRentals', ascending=False)
top_correlated_features = df_corr_sort.head(5).index.tolist()

plt.figure(figsize=(10, 8))
sns.pairplot(data=df_bike, vars=top_correlated_features, hue='BikeRentals', palette='husl')
plt.savefig('correlation_plot.png')
plt.show()

# 첫 번째 그림을 명시적으로 닫습니다.
plt.close()

# 머신러닝 결과 출력
X_train, X_test, y_train, y_test = train_test_split(
    df_bike.drop('BikeRentals', axis=1),
    df_bike['BikeRentals'].values,
    test_size=0.2,
    random_state=42
)

# 결측치를 채우고 랜덤 포레스트 모델을 훈련합니다.
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# 모델 생성
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_imputed, y_train)

# 테스트
predictions = model.predict(X_test_imputed)

#### 7. Machine Learning 결과 출력 및 저장 ####
# 머신러닝 결과를 'machine_learning_results.txt'에 저장합니다.
with open('machine_learning_results.txt', 'w') as f:
    print(f'Machine Learning Results:\n\n', file=f)
    print(f'Predictions: {predictions}\n', file=f)
    print(f'True Values: {y_test}\n', file=f)

# 결과를 로그로도 출력합니다.
print('Machine Learning Results:\n')
print(f'Predictions: {predictions}\n')
print(f'True Values: {y_test}\n')

# 머신러닝 결과를 코랩에서 확인합니다.
try:
    from IPython.display import display
    result_df = pd.read_csv('machine_learning_results.txt', delimiter='\t')
    display(result_df)
except ImportError:
    print("\nMachine Learning Results:")
    print(pd.read_csv('machine_learning_results.txt', delimiter='\t'))

