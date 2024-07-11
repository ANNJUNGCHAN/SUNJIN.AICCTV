import cv2
import os
os.chdir('/code')
from utils import *

setting_path = '/code/setting/'
save_path = '/code/sample/'
Farm = 'BUGUN'
COUNTERS = ["DEAD", "OUT"]

with open(os.path.join(setting_path, Farm + '.json'), 'r') as f :
    json_data = json.load(f)

for house in json_data['CAM_NAME'].values() :
    
    ## 카메라 번호 찾기
    cam_name_dict = json_data['CAM_NAME']
    reverse_cam_name_dict = {v:k for k,v in cam_name_dict.items()}
    cam_no = reverse_cam_name_dict[house]
    
    
    if json_data['Mode'][cam_no] == 'PIG' : # 돼지 카운팅에서만 이용
    
        for counter in COUNTERS :

            rtsp, region_points = SearchParam(setting_path, Farm, house, counter)
            print(Farm, house, cam_no, rtsp)
            
                
            while True :
                
                cap = cv2.VideoCapture(rtsp)
                
                
                try :
                    
                    
                    success, im0 = cap.read()
                    
                    if im0 is None :
                        pass
                    
                    else :
                        break
                
                except :
                    pass
                
            cv2.imwrite(os.path.join(save_path, Farm + "_" + house + "_" + counter + ".jpg"), im0)
    
    else :
        pass