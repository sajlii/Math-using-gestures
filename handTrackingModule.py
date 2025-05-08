import cv2
import mediapipe as mp

class handDetector():
    def __init__(self, mode=False, max_hands=1, model_complexity=1, min_det_conf=0.7, min_tracking_confidence=0.7):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.min_det_conf = min_det_conf
        self.min_tracking_confidence = min_tracking_confidence
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode, self.max_hands, self.model_complexity,
                                        self.min_det_conf, self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLMS, self.mphands.HAND_CONNECTIONS)
        return img
    def findPosition(self,img,draw=True):
        myHand={}
        lmList=[]
        allHands=[]
        if self.results.multi_hand_landmarks:
            for handType,handLMS in zip(self.results.multi_handedness,self.results.multi_hand_landmarks):
               for id,lm in enumerate(handLMS.landmark):
                   h,w,c=img.shape
                   cx,cy,cz= int(lm.x*w),int(lm.y*h),int(lm.z*w)
                   lmList.append([id,cx,cy,cz])
               myHand["lmList"]=lmList
               if handType.classification[0].label=="Right":
                   myHand["type"]="Left"
               else:
                   myHand["type"]="Right"
               allHands.append(myHand)
            if draw:
                cv2.circle(img,(lmList[8][1],lmList[8][2]),5,(255,0,0),cv2.FILLED)
        return allHands,img

    def fingersUp(self, hand):
        fingers = []
        lmList = hand["lmList"]
        handType = hand["type"]

        # Thumb
        if handType == "Right":
            fingers.append(1 if lmList[4][1] > lmList[3][1] else 0)
        else:
            fingers.append(1 if lmList[4][1] < lmList[3][1] else 0)

        # Fingers (index to pinky)
        for tipId in [8, 12, 16, 20]:
            fingers.append(1 if lmList[tipId][2] < lmList[tipId - 2][2] else 0)

        return fingers


def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        ret, frame = cap.read()
        if ret:

            frame = detector.findHands(frame)
            allHands,img=detector.findPosition(frame)
            if allHands:
                print(allHands)
                hand1=allHands[0]
                lmList=hand1["lmList"]
                type=hand1["type"]
                cv2.circle(img, (lmList[4][1], lmList[4][2]), 5, (0, 255, 0), cv2.FILLED)
                fingers=detector.fingersUp(hand1)
                print(fingers)
                print(f"H1={fingers.count(1)}{fingers}")
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('1'):
                break
        else:
            break

if __name__ == "__main__":
    main()
