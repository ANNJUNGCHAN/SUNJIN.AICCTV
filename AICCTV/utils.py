import cv2
import datetime
import os
import json
from ultralytics.solutions import object_counter

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
        region_points.append((int(sublist[0]), int(sublist[1])))
        
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

def MakeCap(rtsp) :
    
    cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    
    return cap, w, h, fps