import cv2 as cv
import tensorflow as tf
import numpy as np


class VideoComp:
    def __init__(self, video_path, model_path, threshold=1.2):
        self.video_path = video_path
        self.cap = cv.VideoCapture(video_path)
        self.model_path = model_path
        self.model = tf.keras.models.load_model(model_path)
        self.threshold = threshold
        self.fps = self.cap.get(cv.CAP_PROP_FPS)
        self.num_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        # self.current_index = 0
        # self.frames_code = []
        # self.time_code = []
        self.dict_time = {}
        self.step = 2

    def get_frame(self, index, size=(128, 128)):
        """Return the frame with the specified index from your video.

        index: index of frame
        size: size of frame
        :return: frame
        """
        self.cap.set(cv.CAP_PROP_POS_FRAMES, index)
        _, frame = self.cap.read()
        frame = cv.resize(frame, size) / 255
        return frame

    def apply_encoder(self, frame):
        """Apply encoder to the frame.

        frame: frame to which will be applied encoder
        :return: vector/vectors
        """
        frame = np.expand_dims(frame, axis=0)
        return self.model.predict(frame)

    @staticmethod
    def compare_encoded_frames(enc_frame1, enc_frame2):
        """Apply encoder to frames and compare the resulting vectors.

        enc_frame1: first encoded frame
        enc_frame2: second encoded frame
        :return: MSE between the encoded frames
        """
        # return tf.keras.losses.MSE(enc_frame1, enc_frame2).numpy()
        return np.sqrt(np.sum((enc_frame1 - enc_frame2) ** 2))

    def compare_part_cap(self, start_index, end_index, threshold=1.2, step=2):
        """Compare video frames from start_index to end_index with the denoted step.
        If error between frames is higher than threshold then add index into frame_code_part.
        Than create time_code_part where the frames' indexes converted into time format 00h.00m.00s.
        Return dictionary with the keys 'frames' for frame_code_part array and 'time' for time_code_part array.

        start_index: index from which to start
        end_index: index which need to finish compare
        threshold: value to compare with error
        step: step for the next index
        :return: dictionary with keys 'frames' and 'time'
        """
        if start_index + step > self.num_frames:
            raise ValueError("Start index is bigger than number os frames plus step")
        elif end_index > self.num_frames:
            print(f'End index is bigger than number of frames minus step. End index now is {self.num_frames}')
            end_index = self.num_frames

        frames_code_part = []
        curr_frame = self.get_frame(start_index)
        curr_encoded_frame = self.apply_encoder(curr_frame)
        for index in range(start_index + step, end_index - step, step):
            next_frame = self.get_frame(index + step)
            next_encoded_frame = self.apply_encoder(next_frame)
            error = self.compare_encoded_frames(curr_encoded_frame, next_encoded_frame)

            if error >= threshold:
                print(f'Error of index {index:} = {error:}')
                frames_code_part.append(index)
            curr_encoded_frame = next_encoded_frame

        time_code_part = [self.frame2time(frame_code) for frame_code in frames_code_part]
        print(time_code_part)
        dict_time = {'time': time_code_part, 'frames': frames_code_part}
        return dict_time

    def compare_cap(self):
        """Compare frames for all video with special step. If error between frames
        is higher than threshold then add index into dictionary for 2 format.
        First format is a frame index the second is time code (00h.00m.00s).

        :return: Return dictionary with keys 'frames' and 'time'.
        """
        self.dict_time = self.compare_part_cap(0, self.num_frames, self.threshold, self.step)

    def display_frame_by_index(self, index, size=(128, 128), wait=True):
        """Display frame by index. With name 'frame number {index}'.

        index: the index of the frame to be displayed
        size: size of displayed frame
        wait: True means the image won't be destroyed
        """
        frame = self.get_frame(index, size)
        cv.imshow(f'frame number {index}', frame)
        if wait:
            cv.waitKey(0)

    def frame2time(self, frame_index):
        sec = frame_index // self.fps
        minute = 0 + sec // 60
        hour = 0 + minute // 60
        sec = sec % 60
        minute = minute % 60
        time = '%02dh.%02dm.%02ds' % (hour, minute, sec)
        return time

    def time2frame(self, hour, minute, sec):
        return (3600 * hour + minute * 60 + sec) * self.fps




