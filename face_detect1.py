import cv2
import mediapipe as mp          # pip install mediapipe

cap = cv2.VideoCapture("img3.jpg")

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
# 정확도
faceDetection = mpFaceDetection.FaceDetection(0.25)

while True:
    success, img = cap.read()
    if success:
        # 미디어파이프
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = faceDetection.process(imgRGB)
        print(results.detections)

        if results.detections:
            for id, detection in enumerate(results.detections):
                #기본함수
                mpDraw.draw_detection(img,detection)
                #print(id, detection)
                #print(detection.score)
                #print(detection.location_data.relative_bounding_box)
                # bboxC = detection.location_data.relative_bounding_box
                # ih, iw, ic = img.shape
                # bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                # cv2.rectangle(img, bbox, (255,0,255),2)

        cv2.imshow("Image", img)
    if cv2.waitKey(20) & 0xFF == 27:
            break