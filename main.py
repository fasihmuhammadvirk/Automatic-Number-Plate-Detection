import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr 
import util

# define constants
model_cfg_path = os.path.join('model', 'cfg', 'yolov3.cfg')
model_weights_path = os.path.join('model', 'weights', 'model.weights')
class_names_path = os.path.join('model', 'class.names')

img_path = './Input Directory/image4.jpg'

# input_dir = './Input Directory'

# for img_name in os.listdir(input_dir):
#     img_path = os.path.join(input_dir,img_name)

# load class names
with open(class_names_path, 'r') as f:
    class_names = [j[:-1] for j in f.readlines() if len(j) > 2]
    f.close()

# load model
net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)

# load image

img = cv2.imread(img_path)

H, W, _ = img.shape

# convert image
blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)

# get detections
net.setInput(blob)

detections = util.get_outputs(net)

# bboxes, class_ids, confidences
bboxes = []
class_ids = []
scores = [] 


for detection in detections:
    # [x1, x2, x3, x4, x5, x6, ..., x85]
    bbox = detection[:4]

    xc, yc, w, h = bbox
    bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

    bbox_confidence = detection[4]

    class_id = np.argmax(detection[5:])
    score = np.amax(detection[5:])

    bboxes.append(bbox)
    class_ids.append(class_id)
    scores.append(score)

# apply nms
bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)

reader = easyocr.Reader(['en'])

# plot

for bbox_, bbox in enumerate(bboxes):
    xc, yc, w, h = bbox

    # cv2.putText(img,
    #             class_names[class_ids[bbox_]],
    #             (int(xc - (w / 2)), int(yc + (h / 2) - 20)),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             7,
    #             (0, 255, 0),
    #             15)
    
    license_plate = img[int(yc - (h / 2)):int(yc + (h / 2)),int(xc - (w / 2)):int(xc + (w / 2)),:].copy()
    
    img = cv2.rectangle(img,
                        (int(xc - (w / 2)), int(yc - (h / 2))),
                        (int(xc + (w / 2)), int(yc + (h / 2))),
                        (0, 255, 0),
                        10)
    gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
    final_plate = cv2.Canny(bfilter, 30, 200) #Edge detection
    
    output = reader.readtext(final_plate)
    
    for out in output:
        text_bbox , text, text_score = out 
        if text_score > 0.4:
            print(text,text_score)   

plt.figure()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.figure()
plt.imshow(cv2.cvtColor(license_plate, cv2.COLOR_BGR2RGB))

plt.show()
