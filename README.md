# 🗺️ 캘리포니아 집값을 예측하는 회귀 문제 성능 향상 아이디어 입니다. 

* 프로젝트 : California Housing Prices
* 주최 : 울산대학교 산업경영공학부 수업 '머신러닝'  
  
캐글에 업로드 되어 있는 California Housing Prices 데이터셋을 이용하여 본교 수업에서 진행한 프로젝트입니다.   
  
데이터는 아래와 같이 정형데이터이며, 7개의 column으로 이루어져 있습니다.   
<img width="726" alt="image" src="https://github.com/CodeofO/School-Term-Project/assets/99871109/19a5ec0c-5684-4847-a5f9-ffed942a5daa">
target 값 : MedVal  
   
   
각 column들의 Correlation matrix   
<img width="981" alt="image" src="https://github.com/CodeofO/School-Term-Project/assets/99871109/6e0686ae-1a85-405e-8d6f-dc23c6db320e">
   
위도와 경도가 target 값과 크게 영향을 미치지 않는 것을 볼 수 있습니다.   
   
하지만 아래 지도 사진을 보면 **'집값이 특정 지역에 몰려있다'** 라는 것을 눈으로 볼 수 있었습니다. 
<img width="998" alt="image" src="https://github.com/CodeofO/School-Term-Project/assets/99871109/ed69c2a8-5960-4379-a373-659af441169c">

따라서 이와 같은 가설을 세울 수 있었습니다. 
**"집값은 샌프란시스코와 같은 특정 지역을 주변으로 높게 형성 되어 있다."**

이 가설을 시험해보고자 다음과 같은 순서를 따라 전처리를 진행하였습니다. 
##### 1) 육안으로 봤을 때 높은 집 값들로 형성되어 있는 위치를 확인합니다.
```
# San Francisco Bay의 좌표
center_lat_sf = 37.7749
center_lon_sf = -122.4194

# seal beach의 좌표
center_lat_sb = 33.65
center_lon_sb = -118.06

# Lake Tahoe의 좌표
center_lat_tahoe = 39.0968
center_lon_tahoe = -120.0324

# More Mesa의 좌표
center_lat_mesa = 34.4348
center_lon_mesa = -119.7910

# Pismo Beach의 좌표
center_lat_pb = 35.1428
center_lon_pb = -120.6413
```
(위도, 경도 출처 : Google)
  
👉 총 5곳의 위치가 선정되었습니다.   
  
##### 2) 모든 sample들과 선정된 5곳과의 거리를 파생변수로 생성하였습니다.
  <img width="993" alt="image" src="https://github.com/CodeofO/School-Term-Project/assets/99871109/e5b5b2b3-976c-4319-916b-a24ece9bfc76">

```
# San Francisco Bay의 좌표
center_lat_sf = 37.7749
center_lon_sf = -122.4194

def sanfrancisco_bay(df): # ✅
    df['sanfrancisco_bay'] = np.sqrt((df['Latitude'] - center_lat_sf)**2 + (df['Longitude'] - center_lon_sf)**2)
    return df

x_train = sanfrancisco_bay(x_train)
x_test = sanfrancisco_bay(x_test)

# seal beach의 좌표
center_lat_sb = 33.65
center_lon_sb = -118.06

def Seal_Beach(df):
    df['Seal_Beach'] = np.sqrt((df['Latitude'] - center_lat_sb)**2 + (df['Longitude'] - center_lon_sb)**2)
    return df

x_train = Seal_Beach(x_train)
x_test = Seal_Beach(x_test)

# Lake Tahoe의 좌표
center_lat_tahoe = 39.0968
center_lon_tahoe = -120.0324

def Lake_Tahoe(df):
    df['Lake_Tahoe'] = np.sqrt((df['Latitude'] - center_lat_tahoe)**2 + (df['Longitude'] - center_lon_tahoe)**2)
    return df

x_train = Lake_Tahoe(x_train)
x_test = Lake_Tahoe(x_test)

# More Mesa의 좌표
center_lat_mesa = 34.4348
center_lon_mesa = -119.7910

def More_Mesa(df):
    df['More_Mesa'] = np.sqrt((df['Latitude'] - center_lat_mesa)**2 + (df['Longitude'] - center_lon_mesa)**2)
    return df

x_train = More_Mesa(x_train)
x_test = More_Mesa(x_test)

# Pismo Beach의 좌표
center_lat_pb = 35.1428
center_lon_pb = -120.6413

def Pismo_Beach(df):
    df['Pismo_Beach'] = np.sqrt((df['Latitude'] - center_lat_pb)**2 + (df['Longitude'] - center_lon_pb)**2)
    return df

x_train = Pismo_Beach(x_train)
x_test = Pismo_Beach(x_test)
```

이렇게 1) 위치 선정과 2) 선정된 위치와의 거리를 파생변수로 생성하였습니다. 

그 결과 모델의 성능은 
```0.44 -> 0.41(RMSE, 평균 : priviate, public)```로 상승하였습니다. (전체 2등)

실제로 만든 파생변수는 모델이 예측하는데 큰 도움을 주었음을 ```feature importance```를 통해 확인하였습니다. 
<img width="1054" alt="image" src="https://github.com/CodeofO/School-Term-Project/assets/99871109/c1ab6791-f064-4ebb-baf1-6cd1fbf6ed66">


### 본 프로젝트를 통해서 그냥 EDA 통해 예측에 도움이 될 방법을 고민하는것이 Data Scientist로서의 기본 역량임을 알게 되었습니다. 
