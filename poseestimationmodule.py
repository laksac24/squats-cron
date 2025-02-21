import cv2
import mediapipe as mp
import math

class poseDetector():
    def __init__(self, smode = False, mcomplexity = 1, slandmarks = True, esegmentation = False,
                 ssegmentation = True, dconfidence = 0.5, tconfidence = 0.5):
        self.smode = smode
        self.mcomplexity = mcomplexity
        self.slandmarks = slandmarks
        self.esegmentation = esegmentation
        self.ssegmentation = ssegmentation
        self.dconfidence = dconfidence
        self.tconfidence = tconfidence
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.smode, self.mcomplexity, self.slandmarks, self.esegmentation,
                                     self.ssegmentation, self.dconfidence, self.tconfidence)

    def findPose(self, img, draw = True):
        rgbFrame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(rgbFrame)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw = True):
        self.lmList=[]
        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw = True):
        if len(self.lmList) > max(p1, p2, p3):
            _, x1, y1 = self.lmList[p1]
            _, x2, y2 = self.lmList[p2]
            _, x3, y3 = self.lmList[p3]

            angle = math.degrees(math.atan2(y3-y2,x3-x2) - math.atan2(y1-y2,x1-x2))
            if angle<0:
                angle+=360

            if draw:
                cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)
                cv2.line(img, (x2, y2), (x3, y3), (255, 255, 255), 3)
                cv2.circle(img, (x1, y1), 5, (255, 255, 255), -1)
                cv2.circle(img, (x1, y1), 10, (255, 255, 255), 2)
                cv2.circle(img, (x2, y2), 5, (255, 255, 255), -1)
                cv2.circle(img, (x2, y2), 10, (255, 255, 255), 2)
                cv2.circle(img, (x3, y3), 5, (255, 255, 255), -1)
                cv2.circle(img, (x3, y3), 10, (255, 255, 255), 2)
                cv2.putText(img, str(int(angle)),(x2-10,y2-10),cv2.FONT_HERSHEY_DUPLEX,1, (0,255,255),2)

            return angle
        else:
            print("Landmarks not detected")

def main():
    cap = cv2.VideoCapture("./Videos/Pushups.mp4")
    detector = poseDetector()
    while True:
        sucess, frame = cap.read()
        if sucess:
            frame = detector.findPose(frame)
            lmList = detector.findPosition(frame)
            print(lmList)
            cv2.imshow("output",frame)
            if cv2.waitKey(30) & 0xFF==ord('q'):
                break
        else:
            break
if __name__ == '__main__':
    main()
