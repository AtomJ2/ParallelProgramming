import argparse
import threading
import queue
import time

from ultralytics import YOLO
import cv2


def fun_thread_read(path_video: str, frame_queue: queue.Queue, event_stop: threading.Event):
    cap = cv2.VideoCapture(path_video)
    ind = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame!")
            break
        frame_queue.put((frame, ind))
        ind += 1
        time.sleep(0.0001)
    event_stop.set()


def fun_thread_write(length: int, fps: int, out_queue: queue.Queue, out_path: str):
    t = threading.current_thread()
    frames = [None] * length
    while getattr(t, "do_run", True):
        try:
            frame, ind = out_queue.get(timeout=1)
            frames[ind] = frame
        except queue.Empty:
            pass
    print("Stopping to write.")

    print("Starting to compose.")
    width, height = frames[0].shape[1], frames[0].shape[0]
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame in frames:
        out.write(frame)
    out.release()


def fun_thread_safe_predict(frame_queue: queue.Queue, out_queue: queue.Queue, event_stop: threading.Event):
    local_model = YOLO(model="yolov8s-pose.pt", verbose=False)
    while True:
        try:
            frame, ind = frame_queue.get(timeout=1)
            results = local_model.predict(source=frame, device='cpu')[0].plot()
            out_queue.put((results, ind))
        except queue.Empty:
            if event_stop.is_set():
                print(f'Thread {threading.get_ident()} final!')
                break


def main(arg):
    threads = []
    frame_queue = queue.Queue(1000)
    out_queue = queue.Queue()
    event_stop = threading.Event()
    video_path = arg.inp
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    thread_read = threading.Thread(target=fun_thread_read, args=(video_path, frame_queue, event_stop,))
    thread_read.start()

    thread_write = threading.Thread(target=fun_thread_write, args=(length, fps, out_queue, arg.out,))
    thread_write.start()

    start_t = time.monotonic()
    for _ in range(arg.th):
        threads.append(threading.Thread(target=fun_thread_safe_predict, args=(frame_queue, out_queue, event_stop,)))

    for thr in threads:
        thr.start()

    for thr in threads:
        thr.join()

    thread_read.join()

    thread_write.do_run = False
    thread_write.join()

    end_t = time.monotonic()
    print(f'Time: {end_t - start_t}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp', type=str, default=None, help='Path to your input video')
    parser.add_argument('--out', type=str, default=None, help='Path to your output video')
    parser.add_argument('--th', type=int, default=1, help='Number of threads to parse on')
    args = parser.parse_args()
    main(args)
