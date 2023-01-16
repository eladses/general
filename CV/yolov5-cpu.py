import cv2
import psutil
import torch
from matplotlib import pyplot as plt

#Model
model = torch.hub.load('yolov5', 'custom', path='yolov5s.pt', source='local') # local repo # custom



def foo(img):
    """
    function to process image
    :param img: an image
    :return: a processed image
    """
    a = pow(2.1, 500)
    return img

def foo2(img):
    """
    function to process image
    :param img: an image
    :return: a processed image
    """
    # Inference
    results = model(img)

    # Results
    print(results)
    # cv2.imshow(img,"bicubic")
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(results.render()[0])

    plt.show()
    # print(results.xywh[0])
    return img


def add_texts(img, fps, current_frame, cpu_percent):
    cv2.putText(img, f"fps: {str(int(fps))}", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, f"frame number: {current_frame}", (75, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, f"cpu: {cpu_percent}%", (75, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


def cpu_utilit(cap_input=0, frame_number=8, func=foo):
    """
    loop on cap and sends it to 'func' till pressed 'q'

    :param cap_input: the camera input (default to 0)
    :param frame_number: in with frame-number-th to activate func
    :param func: a function that gets image to process
    """

    cap = cv2.VideoCapture(cap_input)

    # frame counter
    counter = -1
    cpu = psutil.cpu_percent()

    while True:

        timer = cv2.getTickCount()
        success, img = cap.read()

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        counter = (counter + 1) % frame_number
        # if counter on 0 call func
        if not counter:
            img = func(img)

        cpu=psutil.cpu_percent()
        print(f"current frame: {counter}\t fps:{float('%.2f' % fps)}   \t cpu: {cpu}")

        add_texts(img, fps, counter, cpu)
        cv2.imshow("ori", img)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break


cpu_utilit(frame_number=10,func=foo2)
