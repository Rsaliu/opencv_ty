import cv2
import numpy as np

def detect_eyes(img_gray, classifier):
    eyes_coords=[]
    coords = classifier.detectMultiScale(img_gray, 1.3, 5)# detect eyes
    height = np.size(img_gray, 0) # get face frame height
    width = np.size(img_gray, 1)
    for (x, y, w, h) in coords:
        eyecenter = x + w / 2
        if y > height/2: # pass if the eye is at the bottom
            continue
        else:
            eyes_coords.append((x,y,w,h))
    return eyes_coords

def detect_face(img_gray, classifier):
    faces = classifier.detectMultiScale(img_gray, 1.3, 5)
    if len(faces) > 1:
        biggest_face=(0,0,0,0)
        for face in faces:
            if face[3] > biggest_face[3]:
                biggest_face = face
        print(f"biggest face is: {biggest_face}")
        return biggest_face
    else:
        print(f"single face {faces}")
        return faces

def image_face_and_eye_detector():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 
    img = cv2.imread('human_face.jpg',-1)
    eye_coordinates=[]
    gray_picture = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face_coord = detect_face(gray_picture,face_cascade)
    if len(face_coord) != 0:
        for (x,y,w,h) in face_coord:
            cv2.rectangle(gray_picture,(x,y),(x+w,y+h),(0,225,255),2)
            face_gray = gray_picture[y:y+h, x:x+w]
            eye_coords = detect_eyes(face_gray,eye_cascade)
            if len(eye_coords) != 0:
                for(a,b,c,d) in eye_coords:
                    i=a+x
                    j=b+y
                    eye_coordinates.append((i,j,c,d))
                    cv2.rectangle(gray_picture,(i,j),(i+c,j+d),(0,0,255),2)
                    eye_gray = gray_picture[j:j+d, i:i+c]
    threshold=86
    from numpy import array
    for (x,y,w,h) in eye_coordinates:
        eye_img = gray_picture[y:y+h, x:x+w]
        eye_img=cv2.threshold(eye_img, threshold, 255, cv2.THRESH_BINARY)
        #_, img[y:y+h, x:x+w] = cv2.threshold(gray_picture, threshold, 255, cv2.THRESH_BINARY)
        cv2.namedWindow("eye image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("eye image", 700, 700)
        cv2.imshow("eye image",eye_img[1])
        print("printed eye image")
    #cv2.imshow('image',gray_picture)
    k=cv2.waitKey(0) & 0xFF

    if k == 27:
        cv2.destroyAllWindows()


def video_face_and_eye_detector():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret,img = cap.read()
        gray_picture = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        face_coord = detect_face(gray_picture,face_cascade)
        if len(face_coord) != 0:
            for (x,y,w,h)in face_coord:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,225,255),2)
                face_gray = gray_picture[y:y+h, x:x+w]
                eye_coords = detect_eyes(face_gray,eye_cascade)
                if len(eye_coords) != 0:
                    for(a,b,c,d) in eye_coords:
                        cv2.rectangle(img,(a+x,b+y),(a+c+x,b+d+y),(0,0,255),2)
        cv2.namedWindow("eye image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("eye image", 700, 700)
        cv2.imshow('frame',img)
        if cv2.waitKey(1)&0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_face_and_eye_detector()
    #image_face_and_eye_detector()
