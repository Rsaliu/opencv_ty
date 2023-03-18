import cv2
import numpy as np
import enum
from time import sleep
# Using enum class create enumerations
class ImageType(enum.Enum):
   ONE_DIM = 1
   TWO_DIM = 2

def display_image(image,type_of_image,image_name):
    print("fired display")
    cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(image_name, 700, 700)
    if type_of_image == ImageType.ONE_DIM :
        cv2.imshow(image_name,image) 
    elif type_of_image == ImageType.TWO_DIM:
        cv2.imshow(image_name,image[1])
    else:
        print("Image type unknown")

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
import math
def non_maximal_supression(features,x):
    for f in features:
        distx = f.pt[0] - x.pt[0]
        disty = f.pt[1] - x.pt[1]
        dist = math.sqrt(distx*distx + disty*disty)
        if (f.size > x.size) and (dist<f.size/2):
            return True

def detect_pupil_mser(img_gray, detector):
    keypoints = detector.detect(img_gray)
    print(f"keypoints are {keypoints}")
    keypoints.sort(key = lambda x: -x.size)
    keypoints = [ x for x in keypoints if x.size > 1500]
    reduced_features = [x for x in keypoints if not non_maximal_supression(x)]
    return reduced_features

def detect_pupil(img_gray, detector):
    _, img = cv2.threshold(img_gray, 60, 255, cv2.THRESH_BINARY)
    display_image(img,ImageType.ONE_DIM,"intermediate")
    #img = (img/256).astype('uint8')
    keypoints = detector.detect(img)
    print(f"keypoints are {keypoints}")
    return keypoints

def image_face_and_eye_detector_from_saved_picture():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 
    detector_params = cv2.SimpleBlobDetector_Params()
    detector_params.filterByArea = True
    detector_params.maxArea = 800
    detector_params.minArea = 300
    detector_params.filterByInertia = False
    detector_params.filterByConvexity = False
    pupil_detector = cv2.SimpleBlobDetector_create(detector_params)
    mser_pupil_detector = cv2.MSER_create(400)
    img = cv2.imread('wide_eyes.jpg',-1)
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
    threshold=100
    from numpy import array
    count=0
    for (x,y,w,h) in eye_coordinates:
        eye_img = gray_picture[y:y+h, x:x+w]
        _,eye_img=cv2.threshold(eye_img, threshold, 255, cv2.THRESH_BINARY)
        #_, img[y:y+h, x:x+w] = cv2.threshold(gray_picture, threshold, 255, cv2.THRESH_BINARY)
        count+=1
        name = f"eye image {count}"
        #display_image(eye_img,ImageType.ONE_DIM,name)
        #print("printed eye image")
        pupil_points = detect_pupil(eye_img,pupil_detector)
        #pupil_points = detect_pupil(eye_img,mser_pupil_detector)
        blobbed_pupil = cv2.drawKeypoints(eye_img, pupil_points, outImage=np.array([]), color=(0, 0, 255),flags= cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        display_image(blobbed_pupil,ImageType.ONE_DIM,name)
    #cv2.imshow('image',gray_picture)
    k=cv2.waitKey(0) & 0xFF

    # Close windows when Q is pressed on the keyboard
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
        display_image(img,ImageType.TWO_DIM,"frame")
        if cv2.waitKey(1)&0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #video_face_and_eye_detector()
    image_face_and_eye_detector_from_saved_picture()
