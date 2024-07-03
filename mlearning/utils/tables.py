import pandas as pd

# 주어진 데이터
data = {
    '124022008143984_32_LH_WELD12': {'MARK': 2, 'ROI': 1, 'FILM': 1},
    '124012500221384_9_RH_WELD1': {'ROI': 1, 'MARK': 1, 'FILM': 1},
    '124012500225067_9_RH_WELD1': {'ROI': 1, 'FILM': 1, 'MARK': 1},
    '124012500240851_9_RH_WELD1': {'FILM': 1, 'MARK': 1}
}

# DataFrame 생성
df = pd.DataFrame(data).fillna(0)  # NaN 값을 0으로 채움

# Styler 객체 생성
styler = df.style

# applymap을 사용하여 값이 0인 셀에 배경색을 지정
styler.applymap(lambda x: 'background-color: #ffcccb' if x == 0 else '')

# HTML 형식으로 데이터프레임을 변환
html_table = styler.to_html()

# HTML 파일로 저장
with open('styled_table.html', 'w') as f:
    f.write(html_table)

# 출력하여 확인
print(html_table)
