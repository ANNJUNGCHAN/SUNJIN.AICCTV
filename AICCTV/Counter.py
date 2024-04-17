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

def Detect(q, MODEL_PATH, SETTING_PATH, FARM, HOUSE, COUNTER) :
    
    # 모델 불러오기
    model = YOLO(MODEL_PATH)
    
    # 셋팅값 불러오기
    rtsp, region_points = SearchParam(SETTING_PATH, FARM, HOUSE, COUNTER)
    
    ## 카운터 정의
    counter = MakeCounter(model, region_points)
    
    while True :
    
        try :
            
            ## rtsp 불러오기
            #### 최대한 빠른 재접속을 위해, 모델과 카운터는 앞단에서 불러오고, cap은 뒷단에서 불러온다.
            print(rtsp)
            cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
            
            detect_time = None
            
            frame_count = 0
            
            while cap.isOpened():
                
                success, im0 = cap.read()
                
                if not success:
                    print("Video frame is empty or video processing has been successfully completed.")

                ## 트랙킹
                tracks = model.track(im0, persist=True, show=False, verbose = True)
                boxes, track_ids, im0, in_count, out_count = counter.start_counting(im0, tracks)
                im0 = InsertNowTime(im0)
                frame_count += 1
                
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
                                           10,
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
                        os.rename(save_path, final_save_path)
                        print(f"Video saved: {final_save_path}")
                        
                        ## 텍스트 파일 저장
                        final_counter_txt_name = f"{FARM}_{HOUSE}_{COUNTER}_{start_time.strftime('%Y%m%d_%H%M%S')}_{end_time.strftime('%Y%m%d_%H%M%S')}.txt"
                        final_counter_txt_save_path = os.path.join(SAVE_COUNTER_TXT_PATH, final_counter_txt_name)
                        os.rename(text_save_path, final_counter_txt_save_path)
                        print(f"Counts saved: {final_counter_txt_save_path}")
                        
                        ## 디텍팅 파일 저장
                        final_detect_txt_name = f"{FARM}_{HOUSE}_{COUNTER}_{start_time.strftime('%Y%m%d_%H%M%S')}_{end_time.strftime('%Y%m%d_%H%M%S')}.txt"
                        final_detect_txt_save_path = os.path.join(SAVE_DETECT_TXT_PATH, final_detect_txt_name)
                        os.rename(detect_save_full_path, final_detect_txt_save_path)
                        print(f"Counts saved: {final_detect_txt_save_path}")
                        
                    break

        except Exception as e:
            print(e)
            pass
      
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    ### 해당 2개 값은 설정하는 것이 아닌, 기본값이 파싱되도록 한다.
    ### 만약에 모델에 변경이 생기면, 해당 부분의 경로만 수정해서 전체 적용 되도록 한다.
    parser.add_argument('--model', type=str, default='/Drive/DATACENTER_SSD/AICCTV_ASSET/model/20240319/weights/best.pt', help='insert yolov8 detection model')
    parser.add_argument('--setting', type=str, default='/code/setting', help='insert setting json path')
    ################################################################################
    
    parser.add_argument('--farm', type=str, default='', help='insert farm name')
    parser.add_argument('--house', type=str, default='', help='insert house name')
    parser.add_argument('--counter', type=str, default='', help='insert count type (upper letter, DEAD/OUT)')
    parser.add_argument('--video_path', type=str, default='', help='insert where you save video')
    parser.add_argument('--counter_txt_path', type=str, default='', help='insert where you save counter txt path')
    parser.add_argument('--detect_txt_path', type=str, default='', help='insert where you save counter txt path')
    
    args = parser.parse_args()
    
    # 환경 설정 변수
    MODEL_PATH = args.model
    SETTING_PATH = args.setting
    FARM = args.farm
    HOUSE = args.house
    COUNTER = args.counter
    SAVE_VIDEO_PATH = args.video_path
    SAVE_COUNTER_TXT_PATH = args.counter_txt_path
    SAVE_DETECT_TXT_PATH = args.detect_txt_path
    
    # 큐 객체 생성
    q = queue.Queue()

    # 스레드 생성 및 실행
    t1 = threading.Thread(target=Detect, args=(q, MODEL_PATH, SETTING_PATH, FARM, HOUSE, COUNTER))
    t2 = threading.Thread(target=VideoRecorder, args=(q, SAVE_VIDEO_PATH, SAVE_COUNTER_TXT_PATH, SAVE_DETECT_TXT_PATH, FARM, HOUSE, COUNTER))

    t1.start()
    t2.start()