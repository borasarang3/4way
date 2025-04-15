from ultralytics import YOLO
import cv2
import streamlit as st
import pandas as pd
import plotly.express as px
import timeit

# 모델 로드 / 차후 학습 모델로 변경
model = YOLO("yolo11n.pt")
# model = YOLO("4way-2/4way-main/team_project/runs/detect/train2/weights/best.pt")

# 클래스 그룹 정의
person_id = 0
vehicle_ids = {1, 2, 3, 5, 7}  # 집합(set) 형태로 정의

# 혼잡도 상태 분류 함수
def get_status(count):
    if count <= 30:
        return "Normal"
    elif 31 <= count <= 60:
        return "Warning"
    else:
        return "Danger"

# Streamlit 레이아웃 설정
st.set_page_config(layout="wide")
st.title("🚦 4Way 교차로 분석 시스템 🚦")

# 영상 및 결과 표시 영역
video_area = st.empty()
info_area = st.empty()
chart_area = st.empty()

# 비디오 경로 설정
cap = cv2.VideoCapture("http://210.99.70.120:1935/live/cctv007.stream/playlist.m3u8")

# 프레임 처리 루프
fps = cap.get(cv2.CAP_PROP_FPS)

# 비디오 프레임 처리
while cap.isOpened():
    suc, frame = cap.read()
    
    if not suc:
        st.warning("프레임을 가져올 수 없습니다.")
        break
    
    start_time = timeit.default_timer()

    results = model(frame)    
    annotated_frame = results[0].plot()

    # 탐지된 객체 가져오기
    boxes = results[0].boxes
    cls_list = boxes.cls
    
    # 인파 / 차량 각각 카운팅
    person_count = sum(int(cls) == person_id for cls in cls_list)
    vehicle_count = sum(int(cls) in vehicle_ids for cls in cls_list)
        
    # 상태 평가
    person_status = get_status(person_count)
    vehicle_status = get_status(vehicle_count)
        
    end_time = timeit.default_timer()
    FPS = int(1./(end_time - start_time))
    
    # 바운딩 박스 그리기
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{cls_name} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 프레임 RGB 변환 후 표시
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_area.image(frame, channels="RGB", use_column_width=True)

    # 정보 출력
    info_area.markdown(f"""
    ### 🔍 실시간 혼잡도 정보
    - **사람 수**: {person_count} → `{person_status}`
    - **차량 수**: {vehicle_count} → `{vehicle_status}`
    """)

    # 데이터프레임 생성
    data = pd.DataFrame({
        "구분": ["사람", "차량"],
        "수량": [person_count, vehicle_count],
        "상태": [person_status, vehicle_status]
    })

    # 그래프 시각화
    fig = px.bar(data, x="구분", y="수량", color="상태", text="수량", 
                 title="📊 실시간 혼잡도 현황", color_discrete_map={
                    "Normal": "green", "Warning": "orange", "Danger": "red"
                 })
    chart_area.plotly_chart(fig, use_container_width=True)
        
cap.release()
cv2.destroyAllWindows()

# http://localhost:8501
