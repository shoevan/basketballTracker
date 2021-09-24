import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture("footage/1.mp4")
    initROIFetch = True
    ROIFudgingFactor = 0.4
    shotConditionCounter = 0
    shotCounter = 0
    object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=60, detectShadows=False)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            success, img = cap.read()
            height, width, _ = img.shape
            INIT_ROI_WIDTH = int(width / 2)
            INIT_ROI_HEIGHT = int(height / 2)
            ballInHand = False
            #Recolour image to RGB
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            #Make detection
            results = pose.process(image)

            # Recolour image to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                rHip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                rShoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                rElbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                rWrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                rKnee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                rAnkle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                lHip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                lKnee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                lAnkle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                #print(rShoulder, rElbow, rWrist)
                shootingArmAngle = calculate_angle(rWrist, rElbow, rShoulder)
                rShoulderAngle = calculate_angle(rHip, rShoulder, rElbow)
                lKneeAngle = calculate_angle(lHip, lKnee, lAnkle)
                rKneeAngle = calculate_angle(rHip, rKnee, rAnkle)
                #print(shootingArmAngle)


                if rWrist and initROIFetch:
                    INIT_ROI_HEIGHT = int(rWrist[1] * height + height * ROIFudgingFactor)
                    INIT_ROI_WIDTH = int(rWrist[0] * width - width * ROIFudgingFactor)
                    initROIFetch = False
                mask = object_detector.apply(img)
                _, mask = cv2.threshold(mask, 200, 200, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                roi = grey_image[0:int(height / 2), int(width / 2):width]
                bin_img = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY)
                cntr, heirarchy = cv2.findContours(bin_img[1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 1500:
                        #cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
                        x, y, w, h = cv2.boundingRect(cnt)
                        #print(f"rWrist[0]: {rWrist[0] * width} x:{x} rWrist[1]:{rWrist[1] * height} y:{y}")
                        if rWrist[0] and rWrist[1] and x and y:
                            if 0.90 < rWrist[0] * width / (x + w / 2) < 1.10 and 0.90 < rWrist[1] * height / (y + h / 2) < 1.10 and 35 < shootingArmAngle < 100 and 60 < rShoulderAngle < 120:
                                print(f"Ball shot: {rWrist[0] * width}, {rWrist[1] * height} {(x + w / 2)} {(y + h / 2)}\n"
                                f"Shooting arm angle: {int(shootingArmAngle)}; Right shoulder angle: {int(rShoulderAngle)}\n"
                                f"Left knee angle: {int(lKneeAngle)}; Right knee angle: {int(rKneeAngle)}")
                                shotConditionCounter = 10
                            else:
                                if shotConditionCounter == 1:
                                    shotCounter += 1
                                shotConditionCounter -= 1
                                if shotConditionCounter < 0:
                                    shotConditionCounter = 0
                        #cv2.rectangle(image, (x,y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.circle(image, (x + int(w / 2), y + int(h / 2)), radius=15, color=(0,255,0), thickness=-1)
                for cnt in cntr:
                    #approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
                    #area = cv2.contourArea(cnt)
                    #if ((len(approx) > 8) & (area > 500)):
                    with_contours = cv2.drawContours(image, cnt, -1, (0, 0, 255), 1, offset=(int(width / 2), 0))
                #cv2.imshow("ROI", roi)
                #required_contour = max(with_contours)
                #print(len(required_contour))
                #x,y,w,h = cv2.boundingRect(required_contour)
                #img_copy = cv2.rectangle(image, (x,y), (x + w, y + h), (255, 0, 0), 2)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.putText(image, str(int(shootingArmAngle)),
                            tuple(np.multiply(rElbow, [1920, 1080]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, str(int(rShoulderAngle)),
                            tuple(np.multiply(rShoulder, [1920, 1080]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"Shot Count: {shotCounter}",
                            (300, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("Image", image)
                #cv2.imshow("img_copy", img_copy)
                # cv2.imshow("Mask", mask)
            except Exception as e:
                print(f"Cant find: {e}")
                pass




            if cv2.waitKey(10) & 0xFF == ord("q"):
                break


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
