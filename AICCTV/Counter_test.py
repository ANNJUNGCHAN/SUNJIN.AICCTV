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
    
    # rtsp, region_points = SearchParam(SETTING_PATH, FARM, HOUSE, COUNTER) 
    
    # 셋팅값 불러오기
    _, region_points = SearchParam(SETTING_PATH, FARM, HOUSE, COUNTER) # 테스트 시 적용
    
    rtsp = "/Drive/DATACENTER_HDD/AICCTV_BACKUP_PRE/BUGUN_Dong_1_DEAD_5min.mp4" # 테스트 데이터 넣기
    
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
                time.sleep(1/30) # 테스트 시만 적용
                
                if not success:
                    print("Video frame is empty or video processing has been successfully completed.")

                ## 트랙킹
                tracks = model.track(im0, persist=True, show=False, verbose = False)
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
                    
                # 5분 이상 감지가 없으면 while 루프를 다시 시작합니다.
                if detect_time and (current_time - detect_time > datetime.timedelta(minutes=5)):
                    # 둘다 영일경우
                    if (in_count == 0) and (out_count == 0):
                        pass
                    
                    # 만약 5분이상 되었는데 in_counts와 out_counts가 각각 0이 아닐경우
                    else :
                        print('5분이상이 되어, 카운터를 초기화합니다.')
                        # 카운터 초기화
                        counter.reset_counts()
                    
                    
        except Exception as e :
            print(e)
        
def VideoRecorder(q, SAVE_VIDEO_PATH, SAVE_COUNTER_TXT_PATH, SAVE_DETECT_TXT_PATH, FARM, HOUSE, COUNTER):
    
    print(FARM, HOUSE, COUNTER)
    
    ##################### 초기화
    
    start_time, end_time, frame_count, text_file, detect_text_file, video_writer, counter_save_full_path, detect_save_full_path, video_save_full_path = init_saver(SAVE_VIDEO_PATH, SAVE_COUNTER_TXT_PATH, SAVE_DETECT_TXT_PATH, FARM, HOUSE, COUNTER)

    ##################### 큐를 받아서 처리하는 구간
    while True:
        
        try:
            
            # 큐를 받아오면, video wirter나 counter, detector에 정보를 넣어줌
            ##############################################################################################
            ## Queue 받기
            data = q.get(timeout=60)  # 1분 동안 대기
            
            if start_time is None:
                start_time = datetime.datetime.now()  # 첫 데이터 수신 시간 기록

            boxes, track_ids, im0, in_count, out_count = data
            print(boxes, track_ids, in_count, out_count)

            #print('기록시작')
            #print(video_writer.isOpened())
            video_writer.write(im0)
            #cv2.imwrite(os.path.join('/Drive/DATACENTER_HDD/AICCTV_BACKUP_DONE', f'frame_{frame_count}.jpg'), im0)

            print(os.path.getsize(video_save_full_path))
            #print('기록종료')
            
            # 카운팅이 1올라감. 해당 지표는 데이터가 들어왔는지 안들어왔는지 판단하기 위함임
            frame_count += 1

            # in_count와 out_count를 텍스트 파일에 기록
            text_file.write(f"Frame {frame_count}: In {in_count}, Out {out_count}\n")
            detect_text_file.write(f"Frame {frame_count}: BBOX : {boxes} , TRACK : {track_ids}\n")

        except queue.Empty:
            
            if frame_count > 0 :
                
                # 다 닫기
                video_writer.release()
                text_file.close()
                detect_text_file.close()
                
                # 큐가 비어있다면, 마지막 시간을 찾아내고, 파일을 저장한 후, 초기화를 시킴
                
                end_time = datetime.datetime.now()
                
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
                final_detect_txt_name = f"{FARM}_{HOUSE}_{COUNTER}_{start_time.strftime('%Y%m%d_%H%M%S')}_{end_time.strftime('%Y%m%d_%H%M%S')}.txt"
                final_detect_txt_save_path = os.path.join(SAVE_DETECT_TXT_PATH, final_detect_txt_name)
                os.rename(detect_save_full_path, final_detect_txt_save_path)
                print(f"Counts saved: {final_detect_txt_save_path}")
                
                ## 잠시 쉬기
                time.sleep(10)
                
                # 초기화
                start_time, end_time, frame_count, text_file, detect_text_file, video_writer, counter_save_full_path, detect_save_full_path, video_save_full_path = init_saver(SAVE_VIDEO_PATH, SAVE_COUNTER_TXT_PATH, SAVE_DETECT_TXT_PATH, FARM, HOUSE, COUNTER)
        
            else :
                
                # 다 닫기
                video_writer.release()
                text_file.close()
                detect_text_file.close()
                
                # 1분마다 찍힐거임 (QUEUE에서 1분 기다리니까!)
                print("들어온 데이터 없음")
                # 초기화
                start_time, end_time, frame_count, text_file, detect_text_file, video_writer, counter_save_full_path, detect_save_full_path, video_save_full_path = init_saver(SAVE_VIDEO_PATH, SAVE_COUNTER_TXT_PATH, SAVE_DETECT_TXT_PATH, FARM, HOUSE, COUNTER)

      
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