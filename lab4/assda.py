import threading
import time
import cv2
import logging
import sys
import argparse
import queue


logging.basicConfig(filename='app.log', level=logging.INFO)


class Sensor:
    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")


class SensorX(Sensor):
    """Sensor X"""

    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0

    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data


class SensorCam(Sensor):
    def __init__(self, cam, res):
        if cam == 'default':
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(cam)
        self.cap.set(3, res[0])
        self.cap.set(4, res[1])

    def get(self):
        ret, frame = self.cap.read()

        return frame, ret

    def __del__(self):
        self.cap.release()


class WindowImage:
    def __init__(self, freq):
        self.freq = freq
        cv2.namedWindow("window")

    def show(self, img, s1, s2, s3):
        x = 50
        y = 50
        text1 = f"Sensor 1: {s1}"
        text2 = f"Sensor 2: {s2}"
        text3 = f"Sensor 3: {s3}"
        cv2.putText(img, text1, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, text2, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, text3, (x, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("window", img)

    def __del__(self):
        cv2.destroyWindow("window")


def process(que, sensor):
    while True:
        new_sens = sensor.get()
        if que.empty():
            que.put(new_sens)


def main(args):
    picsize = (int(args.res.split('*')[0]), int(args.res.split('*')[1]))
    sensor1 = SensorX(1)
    sensor2 = SensorX(0.1)
    sensor3 = SensorX(0.01)
    window = WindowImage(args.freq)
    camera = SensorCam(args.cam, picsize)

    if not camera.cap.isOpened():
        logging.info('The camera is turned off.')
        sys.exit()

    que1 = queue.Queue()
    que2 = queue.Queue()
    que3 = queue.Queue()

    thread1 = threading.Thread(target=process, args=(que1, sensor1))
    thread2 = threading.Thread(target=process, args=(que2, sensor2))
    thread3 = threading.Thread(target=process, args=(que3, sensor3))

    thread1.start()
    thread2.start()
    thread3.start()

    sens1 = sens2 = sens3 = 0
    while True:
        if not que1.empty():
            sens1 = que1.get()
        if not que2.empty():
            sens2 = que2.get()
        if not que3.empty():
            sens3 = que3.get()
        sensim, ret = camera.get()
        if not ret or not camera.cap.isOpened() or not camera.cap.grab():
            logging.info('The camera had turned off.')
            break

        window.show(sensim, sens1, sens2, sens3)
        time.sleep(1 / window.freq)

        if cv2.waitKey(1) == ord('q'):
            break

    del window
    # del camera
    # del sens1
    # del sens2
    # del sens3
    # del sensim
    # del que1
    # del que2
    # del que3
    # del thread1
    # del thread2
    # del thread3

    sys.exit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cam', type=str, default='default', help='Camera name')
    parser.add_argument('--res', type=str, default='1280*720', help='Camera resolution')
    parser.add_argument('--freq', type=int, default=60, help='Output frequency')
    args = parser.parse_args()
    main(args)
