import cv2
import queue
import time,os
import threading
q=queue.Queue()

def Receive():
    print("start Reveive")
    cap = cv2.VideoCapture("rtsp://admin:lenovo123@192.168.1.120:554")
    ret, frame = cap.read()
    q.put(frame)
    while ret:
        ret, frame = cap.read()
        q.put(frame)


def Display():
    out_path = '/home/lenovo/wangxf35'
    out = cv2.VideoWriter(os.path.join(out_path,"test.avi"), cv2.VideoWriter_fourcc(*'XVID'), 25, (int(4000), int(3000)))
    print("Start Displaying")
    number = 0
    while True:
        if q.empty() !=True:
            frame=q.get()
            out.write(frame)
            print("number is",number)
            number += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__=='__main__':
    p1=threading.Thread(target=Receive)
    p2 = threading.Thread(target=Display)
    p1.start()
    p2.start()