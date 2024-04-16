## 쓰레드 사용
import threading
import queue
import time

## 객체 감지 필요
import cv2
#import matplotlib.pyplot as plt # 주피터 노트북 이미지 볼 때 사용
from ultralytics import YOLO
from ultralytics.solutions import object_counter

## 기본 패키지
import json
import os
import shutil
#import inspect # 패키지가 어디서 왔는지 볼 때 사용
import datetime

def Detect(q) :
    
    # 환경 설정 변수
    MODEL_PATH = '/Drive/DATACENTER_SSD/AICCTV_ASSET/model/20240319/weights/best.pt'
    SETTING_PATH = '/Drive/DATACENTER_SSD/AICCTV_ASSET/setting'
    FARM = 'BUGUN'
    HOUSE = 'Dong_2'
    COUNTER = 'DEAD'
    
    # 모델 불러오기
    model = YOLO(MODEL_PATH)
    
    # JSON 불러오기
    with open(os.path.join(SETTING_PATH, FARM + '.json'), 'r') as f :
        json_data = json.load(f)
        
    # 필요한 것들 얻기

    ## 카메라 번호 찾기
    cam_name_dict = json_data['CAM_NAME']
    reverse_cam_name_dict = {v:k for k,v in cam_name_dict.items()}
    cam_no = reverse_cam_name_dict[HOUSE]

    ## rtsp 찾기
    rtsp_dict = json_data['RTSP_URL']
    rtsp = rtsp_dict[cam_no]
    # rtsp 백업용
    #rtsp = '/Drive/DATACENTER_HDD/AICCTV_OLD_VIDEO/Dong_2_2024-02-22_09-40-30_09-42-13.avi'

    ## 구역 찾기
    rect = json_data['MULTI_CAM'][cam_no][COUNTER]
    region_points = []
    for sublist in rect:
        region_points.append((int(sublist[0]), int(sublist[1])))
    
    ## 실시간일 경우, Tab 댕기기
    while True :
    
        try :
            ## rtsp 불러오기
            cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
            w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
            
            ## 카운터 정의
            counter = object_counter.ObjectCounter()
            counter.set_args(view_img=False,
                            reg_pts=region_points,
                            classes_names=model.names,
                            draw_tracks=True,
                            view_in_counts = True,
                            view_out_counts = True)
            
            detect_time = None
            
            while cap.isOpened():
                success, im0 = cap.read()
                
                if not success:
                    print("Video frame is empty or video processing has been successfully completed.")
                    #time.sleep(120)
                    break
                    ## 다시 맨 앞 반복문으로 되돌아가서, cap과 counter를 불러오고 다시 시작해

                tracks = model.track(im0, persist=True, show=False, verbose = True)
                boxes, im0, in_count, out_count = counter.start_counting(im0, tracks)
                
                current_time = datetime.datetime.now()
                
                # 객체가 감지되면 현재 시간을 업데이트하고 데이터를 큐에 넣습니다.
                if len(boxes) != 0:
                    detect_time = current_time  # 마지막 감지 시간을 현재 시간으로 설정
                    q.put([boxes, im0, in_count, out_count])
                    
                # 객체가 감지되지 않았지만 마지막 감지 시간으로부터 5분 이내인 경우에도 데이터를 큐에 넣습니다.
                elif detect_time and (current_time - detect_time <= datetime.timedelta(minutes=5)):
                    q.put([boxes, im0, in_count, out_count])
                    
        except Exception as e :
            print(e)
        
def VideoRecorder(q):
    
    SAVE_VIDEO_PATH = '/Drive/DATACENTER_HDD/AICCTV_VIDEO'
    SAVE_TXT_PATH = '/Drive/DATACENTER_SSD/AICCTV_LOG'
    FARM = 'BUGUN'
    HOUSE = 'Dong_2'
    COUNTER = 'DEAD'

    while True:
        try:
            start_time = None
            end_time = None
            frame_count = 0
            
            # 텍스트 파일 초기화
            text_filename = FARM + "_" + HOUSE + "_" + COUNTER +  "_" + "record_temp.txt"
            text_save_path = os.path.join(SAVE_TXT_PATH, text_filename)
            text_file = open(text_save_path, "w")  # 텍스트 파일 열기

            # 파일 이름 설정
            filename =  FARM + "_" + HOUSE + "_" + COUNTER + "_" + "record_temp.avi"
            save_path = os.path.join(SAVE_VIDEO_PATH, filename)

            # 비디오 라이터 초기화
            video_writer = cv2.VideoWriter(save_path,
                                           cv2.VideoWriter_fourcc(*'mp4v'),
                                           15,
                                           (640, 480))

            while True:
                try:
                    data = q.get(timeout=60)  # 1분 동안 대기
                    if start_time is None:
                        start_time = datetime.datetime.now()  # 첫 데이터 수신 시간 기록

                    boxes, im0, in_count, out_count = data
                    print(in_count, out_count)

                    video_writer.write(im0)
                    frame_count += 1

                    # in_count와 out_count를 텍스트 파일에 기록
                    text_file.write(f"Frame {frame_count}: In {in_count}, Out {out_count}\n")

                except queue.Empty:
                    end_time = datetime.datetime.now()
                    video_writer.release()
                    text_file.close()  # 텍스트 파일 닫기
                    
                    if frame_count == 0:
                        os.remove(save_path)
                        os.remove(text_save_path)  # 비디오 파일이 없으면 텍스트 파일도 삭제
                        print("No DATA")
                    else:
                        final_filename = f"{FARM}_{HOUSE}_{COUNTER}_{start_time.strftime('%Y%m%d_%H%M%S')}_{end_time.strftime('%Y%m%d_%H%M%S')}.avi"
                        final_text_filename = f"{FARM}_{HOUSE}_{COUNTER}_{start_time.strftime('%Y%m%d_%H%M%S')}_{end_time.strftime('%Y%m%d_%H%M%S')}.txt"
                        final_save_path = os.path.join(SAVE_VIDEO_PATH, final_filename)
                        final_text_save_path = os.path.join(SAVE_TXT_PATH, final_text_filename)
                        os.rename(save_path, final_save_path)
                        os.rename(text_save_path, final_text_save_path)  # 최종 텍스트 파일 이름 변경
                        print(f"Video saved: {final_save_path}")
                        print(f"Counts saved: {final_text_save_path}")
                        
                    break

        except Exception as e:
            print(e)
            pass
        
        
# 큐 객체 생성
q = queue.Queue()

# 스레드 생성 및 실행
t1 = threading.Thread(target=Detect, args=(q,))
t2 = threading.Thread(target=VideoRecorder, args=(q,))

t1.start()
t2.start()