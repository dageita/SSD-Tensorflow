import cv2
import os


def main():
    camera_url = 'rtsp://admin:lenovo123@192.168.1.120:554'
    out_path = '/home/lenovo/wangxf35'
    out = cv2.VideoWriter(os.path.join(out_path,"test.avi"), cv2.VideoWriter_fourcc(*'XVID'), 25, (int(4000), int(3000)))
    print('\n\n\n\n\n\n\n\n\n hello-1')
    cap = cv2.VideoCapture(camera_url) 
    number = 0
    while(cap.isOpened()):
        print('number is',number)
        ret,frame = cap.read()
        number += 1
        if ret:
            out.write(frame)
        else:
            break
    out.release()
    cap.release()


main()
