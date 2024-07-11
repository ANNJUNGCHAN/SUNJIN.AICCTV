import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import pymysql
from st_aggrid import AgGrid, GridOptionsBuilder
import os
os.chdir('/app')
from utils import *

VIDEO_PATH = "/Drive/DATACENTER_SSD/AICCTV_VIDEO"
DB_SETTING_PATH = "/app/setting/DB.json"
st.set_page_config(layout="wide")

"""
# 한마을 재고 두수 관리 시스템 연구 페이지



"""

host, port, user, password, db = Load_Private_Info(DB_SETTING_PATH)

# connect sql
conn = pymysql.connect(host= host,
                    port = port,
                    user=user,
                    password=password,
                    db=db,
                    charset='utf8')

cur = conn.cursor()
query = "SELECT * FROM AICCTV_DB_DEAD"
cur.execute(query)

conn.commit()

# 데이터 파이썬 데이터프레임으로 만들기
datas = cur.fetchall()
data = pd.DataFrame(datas)
data.columns = ["id","Farm", "House", "COUNTER", "START_TIME", "END_TIME", "IN_COUNT", "OUT_COUNT"]

# AgGrid 설정
gb = GridOptionsBuilder.from_dataframe(data)
gb.configure_selection('single')  # 단일 선택 설정
grid_options = gb.build()

# AgGrid 위젯 추가
response = AgGrid(data, gridOptions=grid_options, enable_enterprise_modules=True)

try :
    # 선택된 행 표시
    Farm= response["selected_rows"]['Farm'].values[0]
    House= response["selected_rows"]['House'].values[0]
    COUNTER= response["selected_rows"]['COUNTER'].values[0]
    START_TIME= response["selected_rows"]['START_TIME'].values[0]
    END_TIME= response["selected_rows"]['END_TIME'].values[0]

    path = TIME_to_VideoPath(Farm,House,COUNTER, START_TIME, END_TIME)
    final_path = os.path.join(VIDEO_PATH, path + ".avi")
    st.text(final_path)

    if os.path.exists(final_path):
        # Streamlit의 다운로드 버튼을 사용하여 파일 제공
        with open(final_path, "rb") as file:
            st.download_button(
                label="Download Video",
                data=file,
                file_name=f"{path}.avi",
                mime="video/avi"
            )
    else:
        st.write("비디오 파일을 찾을 수 없습니다.")
        
except :
    st.write("다운받을 비디오를 선택해주세요!")