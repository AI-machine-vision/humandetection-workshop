import cv2
import os

def vdo_recording():
    len_dir = len(os.listdir('./'))
    global name_path
    name_path = '{}_'.format(len_dir)
    os.mkdir(name_path)
    # input
    cap = cv2.VideoCapture(0)

    # Check whether user selected camera is opened successfully.
    if not (cap.isOpened()):
        print("Could not open video device")
        cap.release()
    else:
        print("Welcome to Human detection")
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        width = int(cap.get(3))
        height = int(cap.get(4))

    size = (width, height)
    result = cv2.VideoWriter('./'+name_path+'/video.mp4', 
                         cv2.VideoWriter_fourcc(*'MP4V'),
                         30, size)
    try:
        count = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if (count/30) < 5:
                cv2.putText(frame, str(5-count//30), (height//2,width//2), font,
                            10, (0, 0, 255), 15, cv2.LINE_AA)
            else:
                result.write(frame)
                cv2.putText(frame, str(count//30-5), (height//2,width//2), font,
                            10, (0, 255, 0), 15, cv2.LINE_AA)
            cv2.imshow('frame',frame)

            count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if (count//30)-5 >= 25:
                break
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        print('Stopped by keyboard interrupt')
        cap.release()
        result.release()

def vdo_extracting():
    vdo_path = './'+name_path+'/video.mp4'
    cap = cv2.VideoCapture(vdo_path)
    save_path_all = './'+name_path+'/images_all/'
    save_path = save_path_all.replace('_all','')
    if not os.path.exists(save_path_all):
        os.mkdir(save_path_all)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    try:
        count = 0
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            cv2.imwrite(save_path_all+'img_{:04d}.jpg'.format(count),frame)
            if count%7==0:
                cv2.imwrite(save_path+'img_{:04d}.jpg'.format(count),frame)
            count += 1
        # When everything done, release the capture
        cap.release()
    except:
        print('Done')
        cap.release()

if __name__ == "__main__":
    vdo_recording()
    vdo_extracting()
