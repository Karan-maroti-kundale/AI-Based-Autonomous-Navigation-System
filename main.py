from ultralytics import YOLO
import cv2, numpy as np, math
from tqdm import tqdm

# ---------- Helpers ----------
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [np.array(vertices, dtype=np.int32)], 255)
    return cv2.bitwise_and(img, mask)

def canny_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def detect_lane_lines(frame):
    h, w = frame.shape[:2]
    edges = canny_edges(frame)
    roi = region_of_interest(edges, [(0,h),(w,h),(w//2,int(h*0.58))])
    lines = cv2.HoughLinesP(roi, 2, np.pi/180, 50, minLineLength=40, maxLineGap=100)
    left, right = [], []
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            if x2 == x1: continue
            slope = (y2-y1)/(x2-x1)
            if abs(slope)<0.5: continue
            intercept = y1 - slope*x1
            if slope<0: left.append((slope, intercept))
            else: right.append((slope, intercept))
    line_img = np.zeros_like(frame)
    lane_center_x = w//2
    def make_points(slope, intercept):
        y1, y2 = h, int(h*0.6)
        x1 = int((y1-intercept)/slope)
        x2 = int((y2-intercept)/slope)
        return x1,y1,x2,y2
    if left:
        ls,li = np.mean(left,axis=0)
        x1,y1,x2,y2 = make_points(ls,li)
        cv2.line(line_img,(x1,y1),(x2,y2),(0,255,0),6)
        lane_center_x = x2
    if right:
        rs,ri = np.mean(right,axis=0)
        x1,y1,x2,y2 = make_points(rs,ri)
        cv2.line(line_img,(x1,y1),(x2,y2),(0,255,0),6)
        lane_center_x = (lane_center_x+x2)//2
    center_offset = lane_center_x - (w//2)
    angle = math.degrees(math.atan2(center_offset, h*0.6))
    return line_img, angle

def yolo_detect(model, frame, conf=0.25):
    results = model.predict(source=frame, conf=conf, verbose=False)
    dets = []
    names = model.model.names
    for r in results:
        if r.boxes is None: continue
        for b in r.boxes.data.cpu().numpy():
            x1,y1,x2,y2,cf,cls = b
            dets.append(((int(x1),int(y1),int(x2),int(y2)), names[int(cls)]))
    return dets

def angle_to_motor_speeds(angle, base=60):
    angle = float(np.clip(angle,-45,45))
    turn = angle/45.0
    left = int(np.clip(base*(1-turn),0,100))
    right = int(np.clip(base*(1+turn),0,100))
    return left,right

# ---------- Main ----------
print("Loading YOLOv8...")
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("input.mp4")
if not cap.isOpened(): raise RuntimeError("input.mp4 not found!")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS) or 24
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output.mp4", fourcc, fps, (w,h))

print("Processing video... Press 'q' to stop early.")
while True:
    ok, frame = cap.read()
    if not ok: break
    lanes_img, angle = detect_lane_lines(frame)
    dets = yolo_detect(model, frame)
    left,right = angle_to_motor_speeds(angle)
    overlay = cv2.addWeighted(frame,1.0,lanes_img,0.8,0)
    for box,name in dets:
        x1,y1,x2,y2 = box
        cv2.rectangle(overlay,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(overlay,name,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
    cv2.putText(overlay,f"Angle:{angle:+.1f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
    cv2.putText(overlay,f"Motors L/R:{left}/{right}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
    out.write(overlay)

    # ---- Live simulation window ----
    cv2.imshow("AI Navigation Simulator", overlay)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Done! See output.mp4")
