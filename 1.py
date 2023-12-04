import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib import font_manager, rc

# 한글 폰트 설정 (Windows 기준, MacOS나 Linux에서는 해당 부분을 각 환경에 맞게 수정)
font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 1) NLPRK_STA.csv 파일 읽기
data = pd.read_csv('NLPRK_STA.csv', encoding='euc-kr')

# 2) 국립공원 육지면적, 탐방객수 컬럼만 선택
X = data[['육지면적', '탐방객수']]

# 3) x축은 '육지면적', y축은 '탐방객수'로 설정하고 22개 국립공원을 mapping
plt.figure(figsize=(10, 6))
plt.scatter(X['육지면적'], X['탐방객수'], c='blue', label='국립공원')
plt.title('국립공원 육지면적 vs 탐방객수')
plt.xlabel('육지면적')
plt.ylabel('탐방객수')

# 4) K-평균 클러스터링 적용 (k=3)
kmeans = KMeans(n_clusters=3, max_iter=100, random_state=42)
X['Cluster'] = kmeans.fit_predict(X)

# 클러스터 중심 표시
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', label='클러스터 중심')

# 결과 그래프 표시
plt.legend()
plt.show()