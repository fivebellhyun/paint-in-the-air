import cv2
import numpy
import os
import main

folderPath = 'header'
list = os.listdir(folderPath)
overlay = []

for imPath in list:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlay.append(image)
header = overlay[0]

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

hand_detect = main.hand_detect()
color = (0, 0, 0)
xp, yp = 0,0

canvas = numpy.zeros((720, 1280, 3), numpy.uint8)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = hand_detect.search_hand(img)
    list = hand_detect.findposition(img, draw=False)
    if len(list)!=0:
        x1, y1 = list[8][1:]
        x2, y2 = list[12][1:]
        fingers = hand_detect.fingersup()
        print(fingers)

        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img,(x1, y1), (x2, y2), (255,255,255), cv2.FILLED)
            if y1 < 100:
                if 150<x1<350:
                    header = overlay[0]
                    color = (0, 0, 255)
                elif 450<x1<700:
                    header = overlay[1]
                    color = (255, 0, 0)
                elif 750<x1<950:
                    header = overlay[2]
                    color = (255, 255, 255)
                elif 1000<x1<1200:
                    header = overlay[3]
                    color = (0, 0, 0)

        elif fingers[1]:
            cv2.circle(img, (x1, y1), 15, color, cv2.FILLED)
            if xp==0 and yp==0:
                xp,yp = x1, y1

            if color == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), color, 75)
                cv2.line(canvas, (xp, yp), (x1, y1), color, 75)

            else:
                cv2.line(img, (xp, yp), (x1, y1), color, 15)
                cv2.line(canvas, (xp, yp), (x1, y1), color, 15)
            xp, yp = x1, y1
        else:
            xp, yp = 0, 0
    imggray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imginv = cv2.threshold(imggray, 50, 255, cv2.THRESH_BINARY_INV)
    imginv = cv2.cvtColor(imginv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imginv)
    img = cv2.bitwise_or(img, canvas)

    img[0:100, 0:1280] = header
    cv2.imshow("Paint on camera", img)
    cv2.imshow("after and", imginv)
    cv2.imshow("after or", canvas)
    cv2.waitKey(1)