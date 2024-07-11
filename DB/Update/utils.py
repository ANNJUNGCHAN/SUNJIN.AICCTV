import re
import pandas as pd
import os
import json
from ast import literal_eval



def Load_Private_Info(SETTING_PATH) :
    
    # JSON 불러오기
    with open(os.path.join(SETTING_PATH), 'r') as f :
        json_data = json.load(f)
        
        host=json_data['host']
        port = json_data['port']
        user=json_data['user']
        password=json_data['password']
        db=json_data['DB']
        
    return host, port, user, password, db

def filter_log(log_list) : 
    
    real_log = []
    
    for log in log_list :
        
        if 'record_temp' in log :
            pass
        
        else :
            real_log.append(log)
            
    return real_log

def find_count(txt_path) :
    
    frames = []

    with open(txt_path, "r") as f:
        for line in f:
            readline = line.strip()

            # 'Frame', 'In', 'Out' 앞의 숫자를 찾는 정규표현식
            pattern = r"(?<=Frame )\d+|(?<=In )\d+|(?<=Out )\d+"

            # 패턴에 일치하는 모든 부분 찾기
            matched_nums = re.findall(pattern, readline)

            frames.append(matched_nums)

    return frames[-1][1], frames[-1][2]

def check_string_format(s):
    pattern = r'^\d{8}'
    if re.match(pattern, s):
        return True
    else:
        return False
    
def extract_house_name_and_date(datename) :
    
    datenamesplit = datename.split("_")
    
    remember_list = []
    # 시간 데이터 위치 확인
    for index, splitelement in enumerate(datenamesplit) :
    
        if check_string_format(splitelement) == True : # 시간 데이터가 리스트에 어디 있는지 체크
            remember = index # 시간 데이터 위치 저장
            remember_list.append(remember)
            
    # 돈사 이름 파악
    house_name = ''
    
    for index, house_name_element in enumerate(datenamesplit[0:remember_list[0]]) :
        
        if index < remember_list[0] - 1 :
            house_name += house_name_element + "_"
        
        else :
            house_name += house_name_element

    return house_name, remember_list[0]

def extract_todb(LOG_PATH, path) :
    
    name = path.replace('.txt','')
    house_name, remember = extract_house_name_and_date(name)
    split_name = name.split("_")
    
    ## 시간 만들기
    start_time = split_name[remember:][0][0:4] + "-" + split_name[remember:][0][4:6] + "-" + split_name[remember:][0][6:8] + " " + split_name[remember:][1][0:2] + ":" + split_name[remember:][1][2:4] + ":" + split_name[remember:][1][4:6]
    end_time = split_name[remember:][2][0:4] + "-" + split_name[remember:][2][4:6] + "-" + split_name[remember:][2][6:8] + " " + split_name[remember:][3][0:2] + ":" + split_name[remember:][3][2:4] + ":" + split_name[remember:][3][4:6]
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    
    ## 카운터 추출
    COUNTER = split_name[remember - 1]
    
    ## Farm 이름 추출
    Farm = house_name.split("_")[0]

    ## 동 이름 추출
    House = house_name.replace(Farm + "_","")
    House = House.replace("_"+ COUNTER,"")
    
    ## incout, outcount 추출
    in_count, out_count = find_count(os.path.join(LOG_PATH, path))
    print('c')
    return Farm, House, COUNTER, start_time, end_time, in_count, out_count

def parse_frames(text):
    # 각 프레임 별로 분리
    frames = re.split(r'(?=Frame \d+:)', text.strip())
    frame_ids = []
    bboxes = []
    tracks = []
    
    for frame in frames:
        if frame.strip():  # 비어있지 않은 프레임만 처리
            # Frame ID 추출
            frame_id = re.search(r'Frame (\d+):', frame)
            if frame_id:
                frame_ids.append(int(frame_id.group(1)))
            
            # BBOX 추출
            bbox = re.search(r'BBOX : tensor\((\[\[.*?\]\])', frame, re.DOTALL)
            if bbox:
                bbox_value = literal_eval(bbox.group(1).replace('[', '[').replace(']', ']'))
                bboxes.append(bbox_value)
            
            # TRACK 추출
            track = re.search(r'TRACK : (\[.*?\])', frame)
            if track:
                track_ids = literal_eval(track.group(1))
                tracks.append(track_ids)
    
    return frame_ids, bboxes, tracks

def read_and_parse_file(file_path):
    # 파일 읽기
    with open(file_path, 'r') as file:
        file_content = file.read()
    
    # 데이터 파싱
    frame_ids, bboxes, tracks = parse_frames(file_content)
    return frame_ids, bboxes, tracks