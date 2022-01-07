#pandas import
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 데이터 분포 확인하기 - train data


## 데이터 살펴보기 - train data
'''
|함수|설명|의견|
|------------|----------|---------|
|`.columns`|columns명 가져오기||
|`.shape`| 형태 확인||
|`.unique()`| 중복값 제거한 데이터 보기|series에서 가능|
|`.value_count()`| 값을 인덱스해서 갯수 보기|series에서 가능|

 ### Train 데이터<br>
   - 891개의 데이터(0 ~ 890)
   - 데이터 타입: int형(5개), float(2개), object(5개)
   - Age(714), Cabin(204), Embarked(889)에 결측치가 존재하는걸로 파악됨.

|Variavle|Definition|Key|
|-------------------|---------------------|--------------------|
|PasserngerID|승객 고유 ID||
|Survived    |생존 여부|( 0 = NO,  1 = YES)|
|Pclascc     |티켓 등급|( 1 = 1st, 2 = 2nd, 3 = 3rd)|
|Name        |승객 이름||
|Sex         |승객 성별||
|Age         |승객 나이||
|SibSp       |타이타닉에 탑승한 형제/자매 수||
|Parch       |타이타닉에 탑승한 부모/자녀 수||
|Tichet      |티켓번호||
|Fare        |운임(티켓요금)||
|Cabin       |객실 번호||
|Embarked    |탑승한 곳(항구)|C = Cherboug, Q = Queenstown, S = Southampton||
'''

train.info()
# 데이터 타입에 object확인 / 데이터 분석을 하려면 수치형으로 변환 필요

train.describe()
# 수치로 된 데이터 요약 / object 타입 데이터는 요약 안됨

## 데이터 보기
'''
|함수|설명|의견|
|------------|----------|---------|
|`.unique()`|중복값 없이 데이터 확인|이름 등 중복값이 없는 데이터에서는 얻을 정보가 마땅히 없는듯...|
|`.value_counts()`|값을 인덱스로 하고 그 개수 보기|항목별 갯수 확인<br> `.unique()` 함수와 같이 사용하면 좋은듯|
|`.describe()`|수치 데이터 특성 보기|숫자로 된 데이터에 사용하면 좋은듯?|
'''

train.head(5)

# PassengerID
# 중복값 없음 / 고유ID

# len(train['PassengerId'].unique())
train['PassengerId'].unique()

# Survived
# 0과 1만 존재
# 0 죽은 사람 549명, 1 산 사람 342명

# value-counts() 항목별 갯수 확인 (각각의 고유값을 가지고 있을 값에는 별로 안좋음)
train['Survived'].value_counts()

# Pclasee
# 1등급 216명, 2등급 184명, 3등급 491명

# sort_index(): 오름차순 정렬
train['Pclass'].value_counts().sort_index()

# Name
# 승객의 이름은 모두 다르다

# sort_index(): 오름차순 정렬
# len(train['Name'].unique())
train['Name'].unique()

# Sex
# 남자 577명, 여자 314명

train['Sex'].value_counts()

# Age
# 가장 어린애는 0.42살(약 5개월), 가장 많은 사람은 80세
# 0.5세는 뭘까?
# 89개의 값: 88개의 나이가 있어 + 1개의 결측치
# 결측치 존재
# 소수점 정리 필요

# train['Age'].value_counts().sort_index()
# train['Age'].describe()
# len(train['Age'].unique())
train['Age'].unique()

# SibSp
# 7종류의 값이 존재(0 ,1, 2, 3, 4, 5, 8)
# 형재 자매가 0명 ~ 최대 8명까지
# 0명: 608명, 1명: 209명, 2명: 28명, 3명: 16명, 4명: 18명, 5명: 5명, 8명: 7명


# train['SibSp'].unique()
# train['SibSp'].value_counts()

# Parch
# 7종류의 값이 존재(0 ,1, 2, 3, 4, 5, 6)
# 형재 자매가 0명 ~ 최대 8명까지
# 0명: 678명, 1명: 118명, 2명: 80명, 3명: 5명, 4명: 4명, 5명: 5명, 6명: 1명


# train['Parch'].unique()
# train['Parch'].value_counts()

# Ticket
# 티켓 번호가 같은 사람 존재
# 891명이 탑승, 691개의 티켓이 존재


# train['Ticket'].value_counts()
train['Ticket'].describe()


# Fare
# 248개의 값이 존재: 248개의 요금 종류
# 최대 43명이 같은 요금 지불
# 0원부터 512.3292원까지 요금 존재

# train['Fare'].unique()
# train['Fare'].value_counts()
train['Fare'].describe()

# Cabin
# 148개의 값: 총 147개의 객실이 존재, 1개의 결측치
# 같은 이름의 객실도 존재

# train['Cabin'].value_counts()
# train['Cabin'].unique()
train['Cabin'].describe()

# Embarked
# 4개의 종류 값이 있음: S, C, Q 3개의 항구 + 1개 결측치
# S항구: 644명, C항구: 168명, Q항구: 77명

# train['Embarked'].unique()
train['Embarked'].value_counts()

# EDA - train data

## 결측치 확인
'''
|함수|설명|의견|
|------------|----------|---------|
|`isna().sum()`|컬럼별로 결측치 갯수 확인|`isnull()`과 차이는?|
|`isnull().sum()`|컬럼별로 결측치 갯수 확인|`isna()`와 차이는?|
|`info()`|||
'''
train.isna().sum()

train.isnull().sum()

train.isna().sum().sum()

# 변수별로 생존자수 사망자 수 확인

# 생존자 342명의 성별 확인
# train['Survived']
# train['Survived'] ==  1
# train[train['Survived'] == 1]
# train[train['Survived'] == 1]['Sex']
train[train['Survived'] == 1]['Sex'].value_counts()

# 사망자 549명의 성별 확인
# train['Survived']
# train['Survived'] == 0
# train[train['Survived'] == 0]
# train[train['Survived'] == 0]['Sex']
train[train['Survived'] == 0]['Sex'].value_counts()

qqq = pd.DataFrame([train[train['Survived'] == 1]['Sex'].value_counts(), train[train['Survived'] == 0]['Sex'].value_counts()], index=['Survived', 'Dead']).transpose()
qqq

train[train['Survived'] == 1]['Sex'].value_counts()

type(train[train['Survived'] == 1]['Sex'].value_counts())

train[train['Survived']==1]['Sex'].value_counts()

print(train[train['Survived']==1]['Sex'].value_counts()[0])
print(train[train['Survived']==1]['Sex'].value_counts()[1])
print(sum(train[train['Survived']==1]['Sex'].value_counts()[:0]))
print(sum(train[train['Survived']==1]['Sex'].value_counts()[:1]))

type(train[train['Survived'] == 1]['Sex'].value_counts())

train[train['Survived']==1]['Sex'].value_counts().size

train[train['Survived']==1]['Sex'].value_counts()['female']

train[train['Survived'] == 1]['Sex'].value_counts().index[1]

'''
데이터 프레임 행 개수
`df.shape[0]` or `len(df.index)` or `len(df.axes[0])` <br>
데이터 프레임 열 개수
`len(df.axes[1])`
'''

survived_count = train[train['Survived'] == 1]['Sex'].value_counts()  # 생존자 카운트
dead_count = train[train['Survived'] == 0]['Sex'].value_counts()      # 사망자 카운트

bar_chart_df = pd.DataFrame([survived_count, dead_count]) # 생존자 수와 사망자 수를 dataFrame으로
bar_chart_df.index = ['Survived', 'Dead']
bar_chart_df

sum(bar_chart_df.iloc[0,0:2])+sum(bar_chart_df.iloc[0,0:1])

# 그래프 만드는 함수
def bar_chart(_variable):
  survived_count = train[train['Survived'] == 1][_variable].value_counts()  # 생존자 카운트
  dead_count = train[train['Survived'] == 0][_variable].value_counts()      # 사망자 카운트

  bar_chart_df = pd.DataFrame([survived_count, dead_count]) # 생존자 수와 사망자 수를 dataFrame으로
  bar_chart_df.index = ['Survived', 'Dead']  # 데이터프레임 index
  # 그래프 생성
  bar_chart_df.plot(kind = 'bar',                            # 그래프 모양은 세로막대 그래프 / barh: 수평 바 그래프
                    stacked = True,                          # 데이터를 누적으로 표현 / 전체 생존자, 사망자 합도 볼 수 있음
                    figsize = (10, 7),                       # 그래프 사이즈 설정(가로10, 세로 7)
                    title='Survived & Dead : ' +_variable,   # 그래프 제목
                    xlabel = 'Survived & Dead',              # x축 이름
                    ylabel = 'number of people',             # y축 이름
                    rot = 0,                                 # x축 글씨 90º돌리기
                    )
  for i in range(len(bar_chart_df.axes[0])):                             # female, male 2가지 요소가 있어서 survived_count.size = 2
    for j in range(len(bar_chart_df.axes[1])):
      plt.text(i,                                                         # x축 좌표(0 ~ 1) / 수치형이 아니라서 첫번째 Survived 가 0 Dead가 1 
            (sum(bar_chart_df.iloc[i,0:j+1])+sum(bar_chart_df.iloc[i,0:j]))/2,                            # y축 좌표(0 ~ 549) / 사망자 수 합이 549
            bar_chart_df.columns[j],                                 #
            horizontalalignment='center',
            verticalalignment='center',
            )


bar_chart('Sex')

bar_chart('SibSp')