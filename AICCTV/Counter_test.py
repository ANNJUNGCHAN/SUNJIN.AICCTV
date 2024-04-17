## 쓰레드 사용
import threading
import queue
import time

## 객체 감지 필요
import cv2
from ultralytics import YOLO

## 기본 패키지
import os
import shutil
import datetime
import argparse

## 생성 패키지
from utils import *

def Detect(q, MODEL_PATH, SETTING_PATH, FARM, HOUSE, COUNTER, rtsp) :
    
    # 모델 불러오기
    model = YOLO(MODEL_PATH)
    
    # 셋팅값 불러오기
    _, region_points = SearchParam(SETTING_PATH, FARM, HOUSE, COUNTER)
    
    ## 카운터 정의
    counter = MakeCounter(model, region_points)
    
    stop_processing = False
    
    while not stop_processing :
    
        try :
            
            ## rtsp 불러오기
            #### 최대한 빠른 재접속을 위해, 모델과 카운터는 앞단에서 불러오고, cap은 뒷단에서 불러온다.
            cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
        
            detect_time = None
            
            while cap.isOpened():
                
                success, im0 = cap.read()
                
                if not success:
                    print("Video frame is empty or video processing has been successfully completed.")
                    stop_processing = True
                    break

                ## 트랙킹
                tracks = model.track(im0, persist=True, show=False, verbose = True)
                boxes, track_ids, im0, in_count, out_count = counter.start_counting(im0, tracks)
                im0 = InsertNowTime(im0)
                
                ## 트랙킹까지 완료한 시점
                current_time = datetime.datetime.now()
                
                # 객체가 감지되면 현재 시간을 업데이트하고 데이터를 큐에 넣습니다.
                if len(boxes) != 0:
                    detect_time = current_time  # 마지막 감지 시간을 현재 시간으로 설정
                    q.put([boxes, track_ids, im0, in_count, out_count])
                    
                # 객체가 감지되지 않았지만 마지막 감지 시간으로부터 5분 이내인 경우에도 데이터를 큐에 넣습니다.
                elif detect_time and (current_time - detect_time <= datetime.timedelta(minutes=5)):
                    q.put([boxes, track_ids, im0, in_count, out_count])
                    
                    
        except Exception as e :
            print(e)
        
def VideoRecorder(q, SAVE_VIDEO_PATH, SAVE_COUNTER_TXT_PATH, SAVE_DETECT_TXT_PATH, FARM, HOUSE, COUNTER):

    while True:
        
        try:
            
            start_time = None
            end_time = None
            frame_count = 0
            
            # 텍스트 파일 설정
            counter_txt_name = FARM + "_" + HOUSE + "_" + COUNTER +  "_" + "record_temp.txt"
            counter_save_full_path = os.path.join(SAVE_COUNTER_TXT_PATH, counter_txt_name)
            text_file = open(counter_save_full_path, "w")  # 텍스트 파일 열기
            
            # 디텍팅 파일 설정
            detect_txt_name = FARM + "_" + HOUSE + "_" + COUNTER +  "_" + "record_temp.txt"
            detect_save_full_path = os.path.join(SAVE_DETECT_TXT_PATH, detect_txt_name)
            detect_text_file = open(detect_save_full_path, "w")  # 디텍팅 파일 열기

            # 비디오 파일 설정
            video_path =  FARM + "_" + HOUSE + "_" + COUNTER + "_" + "record_temp.avi"
            video_save_full_path = os.path.join(SAVE_VIDEO_PATH, video_path)

            # 비디오 라이터 초기화
            video_writer = cv2.VideoWriter(video_save_full_path,
                                           cv2.VideoWriter_fourcc(*'mp4v'),
                                           15,
                                           (640, 480))

            while True:
                
                try:
                    
                    ## Queue 받기
                    data = q.get(timeout=60)  # 1분 동안 대기
                    
                    if start_time is None:
                        start_time = datetime.datetime.now()  # 첫 데이터 수신 시간 기록

                    boxes, track_ids, im0, in_count, out_count = data
                    print(in_count, out_count)
                    print(boxes, track_ids)

                    video_writer.write(im0)
                    frame_count += 1

                    # in_count와 out_count를 텍스트 파일에 기록
                    text_file.write(f"Frame {frame_count}: In {in_count}, Out {out_count}\n")
                    detect_text_file.write(f"Frame {frame_count}: BBOX : {boxes} , TRACK : {track_ids}\n")

                except queue.Empty:
                    
                    end_time = datetime.datetime.now()
                    video_writer.release()
                    text_file.close()  # 텍스트 파일 닫기
                    
                    if frame_count == 0:
                        os.remove(video_save_full_path)
                        os.remove(counter_save_full_path)  # 비디오 파일이 없으면 텍스트 파일도 삭제
                        print("No DATA")
                        
                    else:
                        
                        ## 비디오 파일 저장
                        final_video_name = f"{FARM}_{HOUSE}_{COUNTER}_{start_time.strftime('%Y%m%d_%H%M%S')}_{end_time.strftime('%Y%m%d_%H%M%S')}.avi"
                        final_save_path = os.path.join(SAVE_VIDEO_PATH, final_video_name)
                        os.rename(video_save_full_path, final_save_path)
                        print(f"Video saved: {final_save_path}")
                        
                        ## 텍스트 파일 저장
                        final_counter_txt_name = f"{FARM}_{HOUSE}_{COUNTER}_{start_time.strftime('%Y%m%d_%H%M%S')}_{end_time.strftime('%Y%m%d_%H%M%S')}.txt"
                        final_counter_txt_save_path = os.path.join(SAVE_COUNTER_TXT_PATH, final_counter_txt_name)
                        os.rename(counter_save_full_path, final_counter_txt_save_path)
                        print(f"Counts saved: {final_counter_txt_save_path}")
                        
                        ## 디텍팅 파일 저장
                        final_detect_txt_name = f"{FARM}_{HOUSE}_{COUNTER}_{start_time.strftime('%Y%m%d_%H%M%S')}_{end_time.strftime('%Y%m%d_%H%M%S')}.txt"code/Counter_test.py
                        final_detect_txt_save_path = os.path.join(SAVE_DETECT_TXT_PATH, final_detect_txt_name)
                        os.rename(detect_save_full_path, final_detect_txt_save_path)
                        print(f"Counts saved: {final_detect_txt_save_path}")
                        
                    break

        except Exception as e:
            print(e)
            pass
      
if __name__ == '__main__':
    
    # 환경 설정 변수
    MODEL_PATH = '/Drive/DATACENTER_SSD/AICCTV_ASSET/model/20240319/weights/best.pt'
    SETTING_PATH = '/code/setting'
    FARM = 'BUGUN'
    HOUSE = 'Dong_3_BACK'
    COUNTER = 'DEAD'
    SAVE_VIDEO_PATH = '/Drive/DATACENTER_HDD/AICCTV_BACKUP_VIDEO'
    SAVE_COUNTER_TXT_PATH = '/Drive/DATACENTER_SSD/AICCTV_BACKUP_LOG'
    SAVE_DETECT_TXT_PATH = '/Drive/DATACENTER_SSD/AICCTV_BACKUP_Detect_Log'
    rtsp = '/Drive/DATACENTER_HDD/AICCTV_BACKUP_PRE/BUGUN_Dong_2_FAIL_20240416_230200_20240416_230400.avi'
    
    # 큐 객체 생성
    q = queue.Queue()

    # 스레드 생성 및 실행
    t1 = threading.Thread(target=Detect, args=(q, MODEL_PATH, SETTING_PATH, FARM, HOUSE, COUNTER, rtsp))
    t2 = threading.Thread(target=VideoRecorder, args=(q, SAVE_VIDEO_PATH, SAVE_COUNTER_TXT_PATH, SAVE_DETECT_TXT_PATH, FARM, HOUSE, COUNTER))

    t1.start()
    t2.start()