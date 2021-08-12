import cv2
import mediapipe as mp


class hand_detect():
    def __init__(self, mode=False, maxhand=2, detection_confidence=0.7, tracking_confidence=0.7):
        self.mode = mode
        self.maxhand = maxhand
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.detecthand = mp.solutions.hands
        self.hand = self.detecthand.Hands(self.mode, self.maxhand, self.detection_confidence, self.tracking_confidence)
        self.drawline = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def search_hand(self, img, draw=True):
        imgcolor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hand.process(imgcolor)

        if self.result.multi_hand_landmarks:
            for Lhand in self.result.multi_hand_landmarks:
                if draw:
                    self.drawline.draw_landmarks(img, Lhand, self.detecthand.HAND_CONNECTIONS)
        return img

    def findposition(self, img, handnumber=0, draw=True):
        self.landmarkList = []
        if self.result.multi_hand_landmarks:
            myhand = self.result.multi_hand_landmarks[handnumber]

            for index, landmark in enumerate(myhand.landmark):
                heigh, weigh, z = img.shape
                x = int(landmark.x * weigh)
                y = int(landmark.y * heigh)
                self.landmarkList.append([index, x, y])
                if draw:
                    cv2.circle(img, (x, y), 1, (255, 255, 255), cv2.FILLED)
        return self.landmarkList

    def fingersup(self):
        fingers=[]
        if self.landmarkList[self.tipIds[0]][1] < self.landmarkList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if self.landmarkList[self.tipIds[id]][2] < self.landmarkList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers


def main():
    cap = cv2.VideoCapture(0)
    detect = hand_detect()
    while True:
        success, img = cap.read()
        img = detect.search_hand(img)
        landmarkList = detect.findposition(img)
        if len(landmarkList) != 0:
            print(landmarkList[4])
        cv2.imshow("image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()