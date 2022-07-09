import requests
import cv2
import numpy as np
import time
import telebot
import face_recognition
bot = telebot.TeleBot('1671546396:AAHhcAcuOKFKqM514zg9TLDTgnhyKOJws_Q')

from imutils.video import VideoStream
frame = cv2.imread('',cv2.IMREAD_GRAYSCALE)
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture('http://192.168.137.150:8081')


bryan_image = face_recognition.load_image_file("bryan.jpg")
bryan_face_encoding = face_recognition.face_encodings(bryan_image)[0]

known_face_encoding = [
    bryan_face_encoding,
]
known_face_names = [
    "Bryan"
]

while True: 
    
    _,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3,5)
 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    names = []
    
    for encoding in encodings:
        matches = face_recognition.compare_faces(known_face_encoding,
        encoding)
        name = "Unknown"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                best_match_index = np.argmin(matches)
                name = known_face_names[best_match_index]
                counts[name] = counts.get(name, 1) + 1
            name = max(counts, key=counts.get)
        names.append(name)
        
        for ((x, y, w, h), name) in zip(faces, names):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if matches [0] == True: 
                cv2.imwrite("family.jpg", frame)
                bot.send_photo("732053544", photo=open('family.jpg', 'rb'))
                bot.send_message("732053544", "There is some of your family member detected do you want to speak to them by the speaker? please type /speak and text what doyou want to say")
            else:
                db =  requests.get('https://risetkanta.com/dave/WRG2021/input.php?Status=1')
                cv2.imwrite("guest.jpg", frame)
                bot.send_photo('732053544', photo=open('guest.jpg', 'rb'))
                bot.send_message("732053544", "There is an unknown person detected if you know who's the person type some command :\n/Postman or /thieves or type /speak and text what doyou want to say")
                
    # Give command 
    
        

    cv2.imshow("Is that you", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

