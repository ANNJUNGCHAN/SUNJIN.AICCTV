import os
import json

def TIME_to_VideoPath(FARM,HOUSE,COUNTER,START_TIME, END_TIME) :
    start_time_str = str(START_TIME).split("T")[0].replace("-","") + "_" + str(START_TIME).split("T")[1].replace(":","")
    end_time_str = str(END_TIME).split("T")[0].replace("-","") + "_" + str(END_TIME).split("T")[1].replace(":","")
    path = FARM + "_" + HOUSE + "_" + COUNTER + "_" + start_time_str + "_" + end_time_str
    return path

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