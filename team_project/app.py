from ultralytics import YOLO
import cv2
import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import time
import uuid
import os
# 환경 변수 설정으로 dll 충돌 문제 해결
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

# 모델 로드 / 차후 학습 모델로 변경
# model = YOLO("yolo11n.pt")
model = YOLO("C:/Users/Administrator/Desktop/4way/4way-2/4way-main/team_project/runs/detect/train2/weights/best.pt")

# 클래스 그룹 정의 Yolo 사용시 사람=0 , 차량 1,2,3,5,7
person_id = 5
vehicle_ids = {0, 1, 2, 4, 7}  # 집합(set) 형태로 정의

# 혼잡도 상태 분류 함수
def get_status(count):
    if count >= 15:
        return "매우 혼잡"
    elif count >= 10:
        return "혼잡"
    elif count >= 5:
        return "보통"
    else:
        return "원활"

# Streamlit 레이아웃 설정
st.set_page_config(layout="wide")
st.title("🚦 4Way 교차로 분석 시스템 🚦")

# 영상 및 결과 표시 영역
video_area = st.empty()
alert_area = st.empty()
info_area = st.empty()
info1_col, info2_col, info3_col, info4_col = st.columns([1, 1, 1, 1])
info1 = info1_col.empty()
info2 = info2_col.empty()
info3 = info3_col.empty()
info4 = info4_col.empty()
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
    frame = cv2.resize(frame, (640, 480))
    
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")    
    boxes = results[0].boxes
    cls_list = boxes.cls
    ids_list = boxes.id
    
    # 바운딩 박스 및 객체 추적 리스트
    tracked_objects = []
    for box in boxes:
        cls_id = int(box.cls[0])
        obj_id = -1
        if box.id is not None and len(box.id) > 0:
            obj_id = int(box.id[0])
        bbox = list(map(int, box.xyxy[0]))
        tracked_objects.append((obj_id, cls_id, bbox))
        
    # 경고 메시지 저장 리스트
    alerts = []
    
    # 위험 객체 ID 저장용 집합
    danger_ids = set()
    
    # 안전도 분석
    # 거리 계산 함수
    def get_center(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def euclidean_distance(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5

    # 차량 객체 필터링
    vehicle_objects = [(obj_id, bbox) for obj_id, cls_id, bbox in tracked_objects if cls_id in vehicle_ids]

    # 차량 간 거리 측정
    for i in range(len(vehicle_objects)):
        for j in range(i + 1, len(vehicle_objects)):
            id1, box1 = vehicle_objects[i]
            id2, box2 = vehicle_objects[j]
            dist = euclidean_distance(get_center(box1), get_center(box2))

            if dist < 50:
                msg = f"🚨 [{now_str}] 차량 거리 위험({int(dist)}px): 차량 {id1} ↔ 차량 {id2}"
                if msg not in previous_alerts:
                    alerts.append(msg)
                    previous_alerts.add(msg)
                    danger_ids.update([id1, id2])  # 🚨 위험 차량 ID 저장

    # 보행자 객체 필터링
    person_objects = [(obj_id, bbox) for obj_id, cls_id, bbox in tracked_objects if cls_id == person_id]

    # 보행자와 차량 간 거리 측정
    for p_id, p_box in person_objects:
        for v_id, v_box in vehicle_objects:
            dist = euclidean_distance(get_center(p_box), get_center(v_box))

            if dist < 60:
                msg = f"🚨 [{now_str}] 보행자와 차량 거리 위험({int(dist)}px): 보행자 {p_id} ↔ 차량 {v_id}"
                if msg not in previous_alerts:
                    alerts.append(msg)
                    previous_alerts.add(msg)
                    danger_ids.update([p_id, v_id])  # 🚨 위험 차량 ID 저장
                    
    # 위험 객체의 바운딩 박스 색상 표시
    for box in boxes:
        cls_id = int(box.cls[0])
        obj_id = -1
        if box.id is not None and len(box.id) > 0:
            obj_id = int(box.id[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"ID:{obj_id} {cls_name}"

        # 박스 색상 체계:
        # - 위험: 빨간색
        # - 비위험 보행자: 노란색
        # - 비위험 차량: 초록색
        if obj_id in danger_ids:
            box_color = (0, 0, 255)      # 🔴 빨간색
            text_color = (0, 0, 255) 
        else:
            # 🚶 비위험 보행자 → 노란색
            if cls_id == person_id:
                box_color = (0, 255, 255)  # 🟨 노란색
                text_color = (0, 255, 255)
            # 🚗 비위험 차량 → 초록색
            elif cls_id in vehicle_ids:
                box_color = (0, 255, 0)    # 🟩 초록색
                text_color = (0, 255, 0) 
            else:
                # 기타 클래스 (예외 처리)
                box_color = (255, 0, 0)
                text_color = (255, 255, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
    # 프레임 RGB 변환 후 표시
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_area.image(frame, channels="RGB", width=640) #use_column_width=True)
    
    # 인파 / 차량 각각 카운팅
    person_count = sum(int(cls) == person_id for cls in cls_list)
    vehicle_count = sum(int(cls) in vehicle_ids for cls in cls_list)
        
    # 상태 평가
    person_status = get_status(person_count)
    vehicle_status = get_status(vehicle_count)
                    
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
    info_area.markdown("### 🔍 실시간 교차로 혼잡도 정보")
    info1.metric("🚶 보행자 수", person_count)
    info2.metric("📈 보행자 혼잡도", person_status)
    info3.metric("🚗 차량 수", vehicle_count)
    info4.metric("📈 차량 혼잡도", vehicle_status)

    # 5초마다 꺾은선 그래프 갱신
    if now - last_history_update >= update_interval:
        
        # 타임스탬프 생성
        timestamp = datetime.datetime.now()
 
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
            df_person = history_df[history_df["구분"] == "사람"].sort_values("시간").tail(10)
            df_vehicle = history_df[history_df["구분"] == "차량"].sort_values("시간").tail(10)
            
            # 상태별 색상 매핑
            color_map = {
                "원활": "green",
                "보통": "yellow",
                "혼잡": "orange",
                "매우 혼잡": "red"
            }
            
            # 상태 연속 구간별로 자르기 위한 함수 (마커 추가용)
            def split_by_status(df):
                segments = []
                if df.empty:
                    return segments
                current_status = df.iloc[0]["상태"]
                segment = [df.iloc[0]]
                for i in range(1, len(df)):
                    row = df.iloc[i]
                    if row["상태"] == current_status:
                        segment.append(row)
                    else:
                        segments.append(pd.DataFrame(segment))
                        segment = [row]
                        current_status = row["상태"]
                segments.append(pd.DataFrame(segment))
                return segments
            
            # 보행자 상태별 꺾은선 생성
            fig_person = px.line(df_person, x="시간", y="수량", title="🚶 보행자 혼잡도 추이")
            fig_person.update_traces(mode='lines', line=dict(color="blue"))
            
            # 상태별로 마커 추가 (상태가 바뀐 지점마다 마커)
            added_status = set()
            
            for seg in split_by_status(df_person):
                status = seg.iloc[0]["상태"]
                show_legend = status not in added_status
                fig_person.add_scatter(
                    x=seg["시간"], y=seg["수량"],
                    mode="markers",
                    marker=dict(color=color_map.get(status, "gray"), size=8, symbol="circle"),
                    name=status if show_legend else None,
                    showlegend=show_legend
                )
                added_status.add(status)

            # x축을 5초 간격으로 설정
            fig_person.update_layout(
                xaxis=dict(
                    tickformat="%H:%M:%S",
                    tickangle=45,
                    tickmode="linear",
                    dtick=5 * 1000  # 5초 간격으로 설정 (밀리초 단위)
                )
            )

            # 차량 상태별 꺾은선 생성
            fig_vehicle = px.line(df_vehicle, x="시간", y="수량", title="🚗 차량 혼잡도 추이")
            fig_vehicle.update_traces(mode='lines', line=dict(color="blue"))  # 선은 하나의 색으로 고정

            # 상태별로 마커 추가 (상태가 바뀐 지점마다 마커)
            added_status = set()
            
            for seg in split_by_status(df_vehicle):
                status = seg.iloc[0]["상태"]
                show_legend = status not in added_status
                fig_vehicle.add_scatter(
                    x=seg["시간"], y=seg["수량"],
                    mode="markers",
                    marker=dict(color=color_map.get(status, "gray"), size=8, symbol="circle"),
                    name=status if show_legend else None,
                    showlegend=show_legend
                )
                added_status.add(status)

            # x축을 5초 간격으로 설정
            fig_vehicle.update_layout(
                xaxis=dict(
                    tickformat="%H:%M:%S",
                    tickangle=45,
                    tickmode="linear",
                    dtick=5 * 1000  # 5초 간격으로 설정 (밀리초 단위)
                )
            )

            # 시각화 업데이트
            chart_person_area.plotly_chart(fig_person, use_container_width=True, key=f"line_chart_person_{uuid.uuid4()}")
            chart_vehicle_area.plotly_chart(fig_vehicle, use_container_width=True, key=f"line_chart_vehicle_{uuid.uuid4()}")
            
        last_history_update = now
           
cap.release()
cv2.destroyAllWindows()
