# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict

import cv2

from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator, colors

check_requirements("shapely>=2.0.0")

from shapely.geometry import LineString, Point, Polygon


class ObjectCounter:
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the Counter with default values for various tracking and counting parameters."""

        # Mouse events
        self.is_drawing = False
        self.selected_point = None

        # Region & Line Information
        self.reg_pts = [(20, 400), (1260, 400)]
        self.line_dist_thresh = 15
        self.counting_region = None
        self.region_color = (255, 0, 255)
        self.region_thickness = 5

        # Image and annotation Information
        self.im0 = None
        self.tf = None
        self.view_img = False
        self.view_in_counts = True
        self.view_out_counts = True

        self.names = None  # Classes names
        self.annotator = None  # Annotator
        self.window_name = "Ultralytics YOLOv8 Object Counter"

        # Object counting Information
        self.in_counts = 0
        self.out_counts = 0
        self.count_ids = []
        self.class_wise_count = {}
        self.count_txt_thickness = 0
        self.count_txt_color = (255, 255, 255)
        self.line_color = (255, 255, 255)
        self.cls_txtdisplay_gap = 50
        self.fontsize = 0.6

        # Tracks info
        self.track_history = defaultdict(list)
        self.track_inside_out = defaultdict(list)
        self.track_thickness = 2
        self.draw_tracks = False
        self.track_color = None

        # Check if environment support imshow
        #self.env_check = check_imshow(warn=True)
        self.env_check = True

    def set_args(
        self,
        classes_names,
        reg_pts,
        count_reg_color=(255, 255, 255),
        line_thickness=2,
        track_thickness=2,
        view_img=False,
        view_in_counts=True,
        view_out_counts=True,
        draw_tracks=False,
        count_txt_thickness=3,
        count_txt_color=(255, 255, 255),
        fontsize=0.8,
        line_color=(255, 255, 255),
        track_color=None,
        region_thickness=5,
        line_dist_thresh=15,
        cls_txtdisplay_gap=50,
    ):
        """
        Configures the Counter's image, bounding box line thickness, and counting region points.

        Args:
            line_thickness (int): Line thickness for bounding boxes.
            view_img (bool): Flag to control whether to display the video stream.
            view_in_counts (bool): Flag to control whether to display the incounts on video stream.
            view_out_counts (bool): Flag to control whether to display the outcounts on video stream.
            reg_pts (list): Initial list of points defining the counting region.
            classes_names (dict): Classes names
            track_thickness (int): Track thickness
            draw_tracks (Bool): draw tracks
            count_txt_thickness (int): Text thickness for object counting display
            count_txt_color (RGB color): count text color value
            fontsize (float): Text display font size
            line_color (RGB color): count highlighter line color
            count_reg_color (RGB color): Color of object counting region
            track_color (RGB color): color for tracks
            region_thickness (int): Object counting Region thickness
            line_dist_thresh (int): Euclidean Distance threshold for line counter
            cls_txtdisplay_gap (int): Display gap between each class count
        """
        self.tf = line_thickness
        self.view_img = view_img
        self.view_in_counts = view_in_counts
        self.view_out_counts = view_out_counts
        self.track_thickness = track_thickness
        self.draw_tracks = draw_tracks

        print("Polygon Counter Initiated.")
        self.reg_pts = reg_pts
        self.counting_region = Polygon(self.reg_pts)

        self.names = classes_names
        self.track_color = track_color
        self.count_txt_thickness = count_txt_thickness
        self.count_txt_color = count_txt_color
        self.fontsize = fontsize
        self.line_color = line_color
        self.region_color = count_reg_color
        self.region_thickness = region_thickness
        self.line_dist_thresh = line_dist_thresh
        self.cls_txtdisplay_gap = cls_txtdisplay_gap
        

    def extract_and_process_tracks(self, tracks):
        """Extracts and processes tracks for object counting in a video stream."""

        boxes = []
        track_ids = []
        # Annotator Init and region drawing
        self.annotator = Annotator(self.im0, self.tf, self.names)

        if tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy.cpu()
            clss = tracks[0].boxes.cls.cpu().tolist()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()

            # Extract tracks
            for box, track_id, cls in zip(boxes, track_ids, clss):
                
                #print(box)
                
                # Draw bounding box
                # 박스는 그리지 않기 때문에, 주석처리 했음
                # self.annotator.box_label(box, label=f"{self.names[cls]}#{track_id}", color=colors(int(track_id), True))

                # Store class info
                if self.names[cls] not in self.class_wise_count:
                    if len(self.names[cls]) > 5:
                        self.names[cls] = self.names[cls][:5]
                    self.class_wise_count[self.names[cls]] = {"in": 0, "out": 0}

                # Draw Tracks
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)

                # Draw track trails
                if self.draw_tracks:
                    self.annotator.draw_centroid_and_tracks(
                        track_line,
                        color=self.track_color if self.track_color else colors(int(track_id), True),
                        track_thickness=self.track_thickness,
                    )
                    
                current_position = Point(track_line[-1]) # 현재 어디 좌표인지 찍음
                #print(current_position)
                is_inside_now = self.counting_region.contains(current_position) # 현재 좌표가 폴리곤 안에 있는지 파악
                #print(is_inside_now)
                
                """Update the track state and manage counts based on transitions."""
                last_state = self.track_inside_out[track_id][-1] if self.track_inside_out[track_id] else None
                #print(last_state, self.track_inside_out[track_id][-1], self.in_counts, self.out_counts)
                print(self.track_inside_out[track_id])
                if last_state is None:
                    # 처음 상태 기록
                    new_state = 'inside' if is_inside_now else 'outside'
                    self.track_inside_out[track_id].append(new_state)
                    
                elif (last_state == 'inside' and not is_inside_now):
                    # 내부에서 외부로 이동
                    self.out_counts += 1
                    self.class_wise_count[self.names[cls]]["out"] += 1
                    self.track_inside_out[track_id][-1] = 'outside'  # 상태 업데이트
                    
                elif (last_state == 'outside' and is_inside_now):
                    # 외부에서 내부로 이동
                    self.in_counts += 1
                    self.class_wise_count[self.names[cls]]["in"] += 1
                    self.track_inside_out[track_id][-1] = 'inside'  # 상태 업데이트 
                
                print(self.in_counts, self.out_counts)


                #prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

                # 폴리곤 안에 트레킹 점이 있는가?
                #is_inside = self.counting_region.contains(Point(track_line[-1]))

                # 만약 이전 포지션이 None이 아니고, 폴리곤 안에 트래킹 점이 있으며, 트레킹 ID가 이전에 발견되지 않았다면,
                #if prev_position is not None and is_inside and track_id not in self.count_ids:
                #    self.count_ids.append(track_id) # 우선 트래킹 점을 새로 추가한다
                    
                    ### 우리의 counter와는 안맞음
                    # 그리고, 여기서는 x 좌표만 고려하여,
                    # x 좌표가 오른쪽으로 이동하면 양수, 왼쪽으로 이동하면 음수가 나오는데
                    # 만약 오른쪽으로 이동했다면 현재 좌표가 120 이전 좌표가 100이라고 가정했을 때, 120-100 = 20 으로 양수가 나온다
                    # 이 때, counting_region의 중심 좌표와 이전 좌표의 차를 계산하여 incount인지 outcount인지 유추하는데
                    # 이전좌표가 100이고 중심좌표가 110이라고 한다면, 100에서 120으로 이동한 것이고
                    # 그렇다면 오른쪽으로 이동한 것이니까, 오른쪽으로 이동한 것을 in이라고 가정한다면
                    # 중심좌표 - 이전좌표가 양수가 나와야 in인 것이다.
                    # 즉, 우선 오른쪽으로 이동했는가?를 보는 것이고, 이전 좌표가 중심좌표 대비 왼쪽에 있었는가를 보는 것이다.
                    #if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                    #    self.in_counts += 1
                    #    self.class_wise_count[self.names[cls]]["in"] += 1
                    #else:
                    #    self.out_counts += 1
                    #    self.class_wise_count[self.names[cls]]["out"] += 1
            #return boxes
        
        #else :
            #boxes = []
            #return boxes
        
        ## 수정
        label = "Sunjin Pig Counter \t"

        for key, value in self.class_wise_count.items():
            if value["in"] != 0 or value["out"] != 0:
                if not self.view_in_counts and not self.view_out_counts:
                    #label = None
                    pass
                elif not self.view_in_counts:
                    label += f"{str.capitalize(key)}: IN {value['in']} \t"
                elif not self.view_out_counts:
                    label += f"{str.capitalize(key)}: OUT {value['out']} \t"
                else:
                    label += f"{str.capitalize(key)}: IN {value['in']} OUT {value['out']} \t"

        label = label.rstrip()
        label = label.split("\t")

        if label is not None:
            self.annotator.display_counts(
                counts=label,
                tf=self.count_txt_thickness,
                fontScale=self.fontsize,
                txt_color=self.count_txt_color,
                line_color=self.line_color,
                classwise_txtgap=self.cls_txtdisplay_gap,
            )
            
        return boxes, track_ids
    
    def display_frames(self):
        """Display frame."""
        if self.env_check:
            self.annotator.draw_region(reg_pts=self.reg_pts, color=self.region_color, thickness=self.region_thickness)
            #cv2.namedWindow(self.window_name)
            #if len(self.reg_pts) == 4:  # only add mouse event If user drawn region
            #    cv2.setMouseCallback(self.window_name, self.mouse_event_for_region, {"region_points": self.reg_pts})
            #cv2.imshow(self.window_name, self.im0)
            # Break Window
            #if cv2.waitKey(1) & 0xFF == ord("q"):
            #    return

    def start_counting(self, im0, tracks):
        """
        Main function to start the object counting process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0  # store image
        ## 수정
        boxes, track_ids = self.extract_and_process_tracks(tracks)  # draw region even if no objects

        #if self.view_img:
            #self.display_frames()
        self.display_frames()
        ## 수정
        
        return boxes, track_ids, self.im0, self.in_counts, self.out_counts
    
    def reset_counts(self):
        """Resets the in and out counts to zero."""
        self.in_counts = 0
        self.out_counts = 0
        self.count_ids = []  # Clear tracked IDs to prevent counting errors on reset
        self.class_wise_count = {} # class_wise_count까지 메모리를 지워줘야 이미지에 표시가 안된다.
        print("Counters have been reset.")


if __name__ == "__main__":
    ObjectCounter()