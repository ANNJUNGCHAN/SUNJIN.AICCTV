import cv2
import datetime
import os
import json
from ultralytics.solutions import object_counter
from ultralytics.solutions import region_select_inout_counter

def InsertNowTime(frame) :
    time_stamp = datetime.datetime.now()
    time_stamp = time_stamp.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, time_stamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

def SearchParam(SETTING_PATH, FARM, HOUSE, COUNTER) :
    
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

    ## 구역 찾기
    rect = json_data['MULTI_CAM'][cam_no][COUNTER]
    region_points = []
    for sublist in rect:
        region_points.append((int(float(sublist[0]) * 640), int(float(sublist[1]) * 480)))
        
    return rtsp, region_points

def SearchParamTestVer(SETTING_PATH, FARM, HOUSE, COUNTER) :
    
    # JSON 불러오기
    with open(os.path.join(SETTING_PATH, FARM + "_update" + '.json'), 'r') as f :
        json_data = json.load(f)
        
    # 필요한 것들 얻기

    ## 카메라 번호 찾기
    cam_name_dict = json_data['CAM_NAME']
    reverse_cam_name_dict = {v:k for k,v in cam_name_dict.items()}
    cam_no = reverse_cam_name_dict[HOUSE]

    ## rtsp 찾기
    rtsp_dict = json_data['RTSP_URL']
    rtsp = rtsp_dict[cam_no]

    ## 구역 찾기
    rect = json_data['MULTI_CAM'][cam_no][COUNTER]
    region_points = []
    for sublist in rect:
        region_points.append((int(float(sublist[0]) * 640), int(float(sublist[1]) * 480)))
        
    return rtsp, region_points

def MakeCounter(model, region_points) :
    
    ## 카운터 정의
    counter = object_counter.ObjectCounter()
    
    ## 카운터 설정
    counter.set_args(view_img=False,
                    reg_pts=region_points,
                    classes_names=model.names,
                    draw_tracks=True,
                    view_in_counts = True,
                    view_out_counts = True)
    
    return counter

def MakeInOutCounter(model, region_points) :
    
    ## 카운터 정의
    counter = region_select_inout_counter.ObjectCounter()
    
    ## 카운터 설정
    counter.set_args(view_img=False,
                    reg_pts=region_points,
                    classes_names=model.names,
                    draw_tracks=True,
                    view_in_counts = True,
                    view_out_counts = True)
    
    return counter

def MakeCap(rtsp) :
    
    cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    
    return cap, w, h, fps

def init_saver(SAVE_VIDEO_PATH, SAVE_COUNTER_TXT_PATH, SAVE_DETECT_TXT_PATH, FARM, HOUSE, COUNTER) :
    print("초기화")
    start_time = None
    end_time = None
    frame_count = 0
    
    # 텍스트 파일 설정
    counter_txt_name = FARM + "_" + HOUSE + "_" + COUNTER +  "_" + "record_temp.txt"
    print(counter_txt_name)
    counter_save_full_path = os.path.join(SAVE_COUNTER_TXT_PATH, counter_txt_name)
    text_file = open(counter_save_full_path, "w")  # 텍스트 파일 열기
    
    # 디텍팅 파일 설정
    detect_txt_name = FARM + "_" + HOUSE + "_" + COUNTER +  "_" + "record_temp.txt"
    print(detect_txt_name)
    detect_save_full_path = os.path.join(SAVE_DETECT_TXT_PATH, detect_txt_name)
    detect_text_file = open(detect_save_full_path, "w")  # 디텍팅 파일 열기

    # 비디오 파일 설정
    video_path =  FARM + "_" + HOUSE + "_" + COUNTER + "_" + "record_temp.avi"
    print(video_path)
    video_save_full_path = os.path.join(SAVE_VIDEO_PATH, video_path)

    # 비디오 라이터 초기화
    video_writer = cv2.VideoWriter(video_save_full_path,
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                10, # normal mode
                                #30, #test mode
                                (640, 480))
    
    return start_time, end_time, frame_count, text_file, detect_text_file, video_writer, counter_save_full_path, detect_save_full_path, video_save_full_path