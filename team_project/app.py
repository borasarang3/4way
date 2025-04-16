from ultralytics import YOLO
import cv2
import streamlit as st
import pandas as pd
import plotly.express as px
import timeit
import datetime
import time

# 모델 로드 / 차후 학습 모델로 변경
# model = YOLO("yolo11n.pt")
model = YOLO("C:/Users/Administrator/Desktop/4way/4way-2/4way-main/team_project/runs/detect/train2/weights/best.pt")

# 클래스 그룹 정의 Yolo 사용시 사람=0 , 차량 1,2,3,5,7
person_id = 5
vehicle_ids = {0, 1, 2, 4, 7}  # 집합(set) 형태로 정의

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
alert_area = st.empty()
info_area = st.empty()
chart_area_stick = st.empty()
col1, col2 = st.columns([1, 1])
chart_person_area = col1.empty()
chart_vehicle_area = col2.empty()


# 비디오 경로 설정
cap = cv2.VideoCapture("http://210.99.70.120:1935/live/cctv007.stream/playlist.m3u8")

# 시간 관리 변수 추가
last_history_update = time.time()
last_graph_update = time.time()
last_alert_time = 0
alert_timeout = 3
update_interval = 5

previous_alerts = set()

# 프레임 처리 루프
fps = cap.get(cv2.CAP_PROP_FPS)

# 히스토리 저장용 리스트 추가
history = []

# 비디오 프레임 처리
while cap.isOpened():
    suc, frame = cap.read()
    
    if not suc:
        st.warning("프레임을 가져올 수 없습니다.")
        break
    
    now = time.time()
    now_str = datetime.datetime.now().strftime("%H:%M:%S")
    
    results = model(frame)    
    annotated_frame = results[0].plot()
    
    start_time = timeit.default_timer()

    # 탐지된 객체 가져오기
    boxes = results[0].boxes
    cls_list = boxes.cls
    
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
    
    # 인파 / 차량 각각 카운팅
    person_count = sum(int(cls) == person_id for cls in cls_list)
    vehicle_count = sum(int(cls) in vehicle_ids for cls in cls_list)
        
    # 상태 평가
    person_status = get_status(person_count)
    vehicle_status = get_status(vehicle_count)
    
    # 안전도 분석
    # 거리 계산 함수
    def get_center(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def euclidean_distance(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5
    
    # 경고 메시지 저장 리스트
    alerts = []

    # 차량 박스만 추출
    vehicle_boxes = [box.xyxy[0].tolist() for box in boxes if int(box.cls[0]) in vehicle_ids]

    # 차량 간 거리 측정
    for i in range(len(vehicle_boxes)):
        for j in range(i + 1, len(vehicle_boxes)):
            center1 = get_center(vehicle_boxes[i])
            center2 = get_center(vehicle_boxes[j])
            dist = euclidean_distance(center1, center2)

            if dist < 50:
                msg = f"🚨 차량 간 거리 위험 ({int(dist)}px): 차량{i+1} ↔ 차량{j+1} at {now_str}"
                if msg not in previous_alerts:
                    alerts.append(msg)
                    previous_alerts.add(msg)

    # 보행자 박스만 추출
    person_boxes = [box.xyxy[0].tolist() for box in boxes if int(box.cls[0]) == person_id]

    # 보행자와 차량 간 거리 측정
    for p_idx, p_box in enumerate(person_boxes):
        for v_idx, v_box in enumerate(vehicle_boxes):
            p_center = get_center(p_box)
            v_center = get_center(v_box)
            dist = euclidean_distance(p_center, v_center)

            if dist < 60:
                msg = f"🚨 보행자와 차량 간 거리 위험 ({int(dist)}px): 보행자{p_idx+1} ↔ 차량{v_idx+1} at {now_str}"
                if msg not in previous_alerts:
                    alerts.append(msg)
                    previous_alerts.add(msg)
                    
    # 최신 알림 시간 갱신
    if alerts:
        last_alert_time = now
                    
    # 경고 메시지 표시 또는 제거
    if alerts:
        alert_text = "### ⚠️ **실시간 안전 경고**\n"
        alert_text += "\n".join([f"- {msg}" for msg in alerts])
        alert_area.markdown(alert_text)
    elif now - last_alert_time < alert_timeout:
        # 이전 알림 이후 경과 시간이 timeout 이내면 유지
        pass
    else:
        alert_area.empty()
        previous_alerts.clear()  # timeout 이후 이전 알림 기록도 초기화

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

    # 실시간 그래프 (막대그래프)
    fig = px.bar(data, x="구분", y="수량", color="상태", text="수량", 
                 title="📊 실시간 혼잡도 현황", color_discrete_map={
                    "Normal": "green", "Warning": "orange", "Danger": "red"
                 })
    chart_area_stick.plotly_chart(fig, use_container_width=True)
    
    # 5초마다 꺾은선 그래프 갱신
    if now - last_history_update >= update_interval:
        
        # 타임스탬프 생성
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
 
        # 현재 시점 정보 저장
        history.append({
            "시간": timestamp,
            "구분": "사람",
            "수량": person_count,
            "상태": person_status
        })
        history.append({
            "시간": timestamp,
            "구분": "차량",
            "수량": vehicle_count,
            "상태": vehicle_status
        })
    
        # 히스토리 개수 제한 / 사람/차량 각각 100개씩
        history = history[-200:]
    
        # 히스토리 데이터프레임 변환 및 시간순 정렬
        history_df = pd.DataFrame(history)
        history_df = history_df.sort_values(by=["시간", "구분"])
        
        # 추이 시각화 (꺾은선 그래프)
        if len(history_df) >= 4:
            # 사람 / 차량 데이터 분리
            df_person = history_df[history_df["구분"] == "사람"]
            df_vehicle = history_df[history_df["구분"] == "차량"]
            
            # 각각 꺾은선 그래프 생성
            fig_person = px.line(df_person, x="시간", 
                                y="수량", color="상태",
                                title="👤 사람 혼잡도 추이", 
                                color_discrete_map={
                                "Normal": "green", "Warning": "orange", "Danger": "red"
                                },
                                markers=True)
            
            # 선 추가
            fig_person.add_traces(
                px.line(df_person, x="시간", y="수량").data
            )
            
            fig_vehicle = px.line(df_vehicle, x="시간", 
                                y="수량", color="상태",
                                title="🚗 차량 혼잡도 추이",
                                color_discrete_map={
                                "Normal": "green", "Warning": "orange", "Danger": "red"
                                },
                                markers=True)
            
            fig_vehicle.add_traces(
            px.line(df_vehicle, x="시간", y="수량").data
            )

            # 고정된 영역에 업데이트
            chart_person_area.plotly_chart(fig_person, use_container_width=True)
            chart_vehicle_area.plotly_chart(fig_vehicle, use_container_width=True)
            
        last_history_update = now
           
cap.release()
cv2.destroyAllWindows()

# http://localhost:8501
