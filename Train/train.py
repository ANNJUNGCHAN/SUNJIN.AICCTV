
# 도커 밖에서 돌리기
# docker exec -it  AICCTV_train /opt/conda/bin/python /code/Train/train.py
from ultralytics import YOLO

# Load a model (build a new model from YAML)
# model = YOLO('yolov8x.pt')    # 최초의 모델
model = YOLO('yolov8x.yaml')  # 빈 공양식 모델, 다음 경로 참조 /usr/src/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml
# model = YOLO('/Drive/MODEL/20240313/weights/best.pt')  # load a pretrained model (recommended for training)
# model = YOLO('/Drive/BIOSECURITY_MODEL/bio_20240326_a/weights/best.pt')  # load a pretrained model (recommended for training)
#model = YOLO('/Drive/DATACENTER_SSD/AICCTV_ASSET/model/20240624_trial_2/weights/best.pt')  # load a pretrained model (recommended for training)
#/mnt

# Train the model
results = model.train(  data     = '/code/Train/AICCTV_train.yaml',                 # 사용할 데이터
                        epochs   =  40,                          # 에포크
                        patience =  40,                         # 
                        imgsz    =  640,                         # 이미지크기
                        # batch    = 38,                            #
                        batch    = -1,                            #
                        save     = True,                        #
                        save_period = 10,                        #
                        pretrained = True,                       #
                        project  = '/Drive/DATACENTER_SSD/AICCTV_ASSET/model',   # 프로젝트 경로
                        # mosaic = 0,
                        # scale  = 0,
                        # degrees = 180,
                        freeze = None,
                        flipud = 0.5,
                        name = '20240711_trial_1'               # 
                    )