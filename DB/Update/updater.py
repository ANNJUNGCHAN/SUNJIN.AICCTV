import os
os.chdir("/Update")
import pymysql
import time
import shutil

from utils import *

while True :

    try :
        LOG_PATH = "/Drive/SSD/AICCTV_LOG"
        VIDEO_PATH = "/Drive/SSD/AICCTV_VIDEO"
        SETTING_PATH = '/Update/config.json'
        LOG_UPDATED_PATH = '/Drive/SSD/AICCTV_UPDATE_LOG'
        VIDEO_UPDATED_PATH = '/Drive/SSD/AICCTV_UPDATE_VIDEO'

        log_list = os.listdir(LOG_PATH)
        log_list = filter_log(log_list)

        host, port, user, password, db = Load_Private_Info(SETTING_PATH)
        
        # connect sql
        conn = pymysql.connect(host= host,
                            port = port,
                            user=user,
                            password=password,
                            db=db,
                            charset='utf8')

        cursor = conn.cursor()
        print(cursor)

        print("DB 연결 완료")

        if len(log_list) == 0 :
            print(log_list)
            time.sleep(60*60) # 한시간
            print('대기')
            pass

        else :
            
            print
            
            for path in log_list :
                
                print(path)
                
                Farm, House, COUNTER, start_time, end_time, in_count, out_count = extract_todb(LOG_PATH, path)
                

                
                if COUNTER == 'OUT' :
                    print("OUT 시작")
                
                    # 만약, 둘다 0이면 비디오와 로그 모두 제거
                    if (in_count == '0') and (out_count == '0') :
                        shutil.move(os.path.join(VIDEO_PATH, path.replace('.txt','.avi')), os.path.join(VIDEO_UPDATED_PATH, path.replace('.txt','.avi')))
                        shutil.move(os.path.join(LOG_PATH, path),os.path.join(LOG_UPDATED_PATH, path))
                    
                    # 만약, 둘 중 하나가 0이 아니면, 로그는 기록으로 올린 후, 로그만 제거
                    else :
                        
                        ### INSERT DB 달라지는거 체크
                        query = """
                            INSERT INTO AICCTVDB (Farm, House, Counter, start_time, end_time, incount, outcount)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """
                        
                        print(Farm, House, COUNTER, start_time, end_time, in_count, out_count)
                            
                        cursor.execute(query, (Farm, House, COUNTER, start_time, end_time, int(in_count), int(out_count)))

                        # end
                        conn.commit()
                        
                        shutil.move(os.path.join(LOG_PATH, path),os.path.join(LOG_UPDATED_PATH, path))
                
                elif COUNTER == 'Polygon' :
                    print("DEAD 시작")
                    
                    # 만약, 둘다 0이면 비디오와 로그 UPDATED_PATH로 보내기
                    if (in_count == '0') and (out_count == '0') :
                        shutil.move(os.path.join(VIDEO_PATH, path.replace('.txt','.avi')), os.path.join(VIDEO_UPDATED_PATH, path.replace('.txt','.avi')))
                        shutil.move(os.path.join(LOG_PATH, path),os.path.join(LOG_UPDATED_PATH, path))
                    
                    # 만약, 둘 중 하나가 0이 아니면, 로그는 기록으로 올린 후, 로그만 제거
                    else :
                        
                        ### INSERT DB 달라지는거 체크
                        
                        query = """
                            INSERT INTO AICCTV_DB_DEAD (Farm, House, Counter, start_time, end_time, incount, outcount)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """
                        
                        print(Farm, House, COUNTER, start_time, end_time, in_count, out_count)
                            
                        cursor.execute(query, (Farm, House, COUNTER, start_time, end_time, int(in_count), int(out_count)))
                        # end
                        conn.commit()
                        
                        shutil.move(os.path.join(LOG_PATH, path),os.path.join(LOG_UPDATED_PATH, path))
                        
    except Exception as e :
        print(e)