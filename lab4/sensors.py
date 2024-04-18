import argparse
import multiprocessing
import time
import cv2
import logging
import sys


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
        # if not ret or not self.cap.isOpened() or not self.cap.grab():
        #     cv2.destroyAllWindows()
        #     logging.info('The camera was turned off.')
        #     sys.exit()

        return frame

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

    que1 = multiprocessing.Queue()
    que2 = multiprocessing.Queue()
    que3 = multiprocessing.Queue()

    process1 = multiprocessing.Process(target=process, args=(que1, sensor1))
    process2 = multiprocessing.Process(target=process, args=(que2, sensor2))
    process3 = multiprocessing.Process(target=process, args=(que3, sensor3))

    process1.start()
    process2.start()
    process3.start()

    sens1 = sens2 = sens3 = 0
    while True:
        ret, frame = camera.cap.read()
        if not ret or not camera.cap.isOpened() or not camera.cap.grab():
            logging.info('The camera was turned off.')
            process1.terminate()
            process2.terminate()
            process3.terminate()
            sys.exit()
        if not que1.empty():
            sens1 = que1.get()
        if not que2.empty():
            sens2 = que2.get()
        if not que3.empty():
            sens3 = que3.get()
        sensim = camera.get()

        window.show(sensim, sens1, sens2, sens3)
        time.sleep(1 / args.freq)

        if cv2.waitKey(1) == ord('q'):
            break

    process1.terminate()
    process2.terminate()
    process3.terminate()
    process1.join()
    process2.join()
    process3.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam', type=str, default='default', help='Camera name')
    parser.add_argument('--res', type=str, default='1280*720', help='Camera resolution')
    parser.add_argument('--freq', type=int, default=60, help='Output frequency')
    args = parser.parse_args()
    main(args)
