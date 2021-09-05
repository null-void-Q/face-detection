import cv2
import numpy as np
import os
def main():

    inputDir='' #  dir of videos folder
    outputDir='./'


    vids = getVidsInDir(inputDir)

    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    st= cv2.getTickCount()
    print('#'*30,'\n\n')
    #
    for vid in vids:
        print('\nProcessing Video:',vid)
        if os.path.exists(outputDir+vid.split('/')[-1]):
            continue
        s= cv2.getTickCount()
        extractFaceFromVideo(vid,detector,outputDir+vid.split('/')[-1])
        print((cv2.getTickCount()-s)/cv2.getTickFrequency())
    #
    print((cv2.getTickCount()-st)/cv2.getTickFrequency())

def extractFaceFromVideo(video,detector,outvideoPath,dim=(341,256)):
    cap = cv2.VideoCapture(video)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file ",video)
    
    out = cv2.VideoWriter(outvideoPath,cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS),dim)
    while(cap.isOpened()):
        
        ret, frame = cap.read()
        if ret == True:
            frame =  cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame,(640,480))
            detectedFaces = detector.detectMultiScale(frame,1.4, 3)
            if len(detectedFaces) == 0:
                continue
            face = extractFace(frame,detectedFaces[0])
            try:
                face=cv2.cvtColor(face,cv2.COLOR_RGB2BGR)
                face = cv2.resize(face,dim)
                out.write(face)
            except:
                print('error - ',video)
        else: 
            break
    cap.release()
    out.release()

def getVidsInDir(dir):
    vids = []
    for path, directories, files in os.walk(dir):
        for file in files:
            if file.find('.mp4')!=-1:
                vids.append(os.path.join(path, file))
    return vids     

def extractFace(img,box):#optimize it // align face
    return np.copy(img[box[1]: box[1] + box[3], box[0]: box[0] + box[2]])

def getMaxDetection(detection):
    maxDetection = detection[0]
    for d in detection:
        if d['confidence'] > maxDetection['confidence']:
            maxDetection = d
    return maxDetection        
    
def displayFrameWithAnnotation(img,detections):
    for face in detections:
        bounding_box = face['box']
        keypoints = face['keypoints']

        cv2.rectangle(img,
                    (bounding_box[0], bounding_box[1]),
                    (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                    (0,155,255),
                    2)
        cv2.circle(img,(keypoints['left_eye']), 2, (0,155,255), 2)
        cv2.circle(img,(keypoints['right_eye']), 2, (0,155,255), 2)
        cv2.circle(img,(keypoints['nose']), 2, (0,155,255), 2)
        cv2.circle(img,(keypoints['mouth_left']), 2, (0,155,255), 2)
        cv2.circle(img,(keypoints['mouth_right']), 2, (0,155,255), 2)
    cv2.imshow('-',img)
    cv2.waitKey()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()


