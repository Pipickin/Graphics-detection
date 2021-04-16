import cv2 as cv
import tensorflow as tf
import numpy as np


class VideoComp:
    """This class is used for graphics detection in media content"""
    def __init__(self, video_path: str, model_path: str,
                 threshold: float = 1.2, step: int = 2) -> None:
        """Initialize class object.

        :param video_path: path to the video
        :param model_path: path to the model's folder
        :param threshold: value to compare with error
        :param step: step for the next index in comparing
        :return: initialize class object
        :rtype: None
        """
        self.cap = cv.VideoCapture(video_path)
        self.model = tf.keras.models.load_model(model_path)
        self._threshold = threshold
        self.fps = self.cap.get(cv.CAP_PROP_FPS)
        self.num_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.dict_time = {}
        self._step = step

    def get_frame(self, index: int, size: tuple = (128, 128)) -> np.ndarray:
        """Return the frame with the specified index from your video.

        :param index: index of frame
        :param size: size of frame
        :return: frame
        :rtype: np.ndarray
        """
        self.cap.set(cv.CAP_PROP_POS_FRAMES, index)
        _, frame = self.cap.read()
        frame = cv.resize(frame, size) / 255
        return frame

    def apply_encoder(self, frame: np.ndarray) -> np.ndarray:
        """Apply encoder to the frame.

        :param frame: frame to which will be applied encoder
        :return: vector/vectors
        :rtype: np.ndarray
        """
        frame = np.expand_dims(frame, axis=0)
        return self.model.predict(frame)

    @staticmethod
    def compare_encoded_frames(first_enc_frame: np.ndarray,
                               second_enc_frame: np.ndarray) -> np.float64:
        """Apply encoder to frames and compare the resulting vectors.

        :param first_enc_frame: first encoded frame
        :param second_enc_frame: second encoded frame
        :return: MSE between the encoded frames
        :rtype: np.float64
        """
        error = np.sqrt(np.sum((first_enc_frame - second_enc_frame) ** 2))
        # error = tf.keras.losses.MSE(first_enc_frame, second_enc_frame).numpy()
        return error

    def compare_part_cap(self, start_index: int, end_index: int,
                         threshold: float = 1.2, step: int = 2) -> dict:
        """Compare video frames from start_index to end_index with the denoted step.
        If error between frames is higher than threshold then add index into frame_code_part.
        Than create time_code_part where the frames' indexes converted into time format 00h.00m.00s.
        Return dictionary with the keys 'frames' for frame_code_part array and 'time' for time_code_part array.

        :param start_index: index from which to start
        :param end_index: index which need to finish compare
        :param threshold: value to compare with error
        :param step: step for the next index
        :return: dictionary with keys 'frames' and 'time'
        :rtype: dict
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

    def compare_cap(self) -> None:
        """Compare frames for all video with special step. If error between frames
        is higher than threshold then add index into dictionary for 2 format.
        First format is a frame index the second is time code (00h.00m.00s).

        :return: Sets the dictionary with keys 'frames' and 'time' to self.dict_time.
        :rtype: None
        """
        self.dict_time = self.compare_part_cap(0, self.num_frames, self._threshold, self._step)

    def display_frame_by_index(self, index: int, size: tuple = (128, 128),
                               wait: bool = True, dynamic: bool = False) -> None:
        """Display frame by index. With name 'frame number {index}'.

        :param index: the index of the frame to be displayed
        :param size: size of displayed frame
        :param wait: True means the image won't be destroyed
        :param dynamic: True means the image can be resized
        :return: displayed frame
        :rtype: None
        """
        if dynamic:
            cv.namedWindow(f'frame number {index}', cv.WINDOW_NORMAL)
        frame = self.get_frame(index, size)
        cv.imshow(f'frame number {index}', frame)
        if wait:
            cv.waitKey(0)

    def frame2time(self, frame_index: int) -> str:
        """Convert frame index into time format.

        :param frame_index: int
        :return: time converted from frame index
        :rtype: str
        """
        sec = frame_index // self.fps
        minute = 0 + sec // 60
        hour = 0 + minute // 60
        sec = sec % 60
        minute = minute % 60
        time = '%02dh.%02dm.%02ds' % (hour, minute, sec)
        return time

    def time2frame(self, hour: int, minute: int, sec: int) -> int:
        """Convert time format into frame index.

        :param hour: number of hours
        :param minute: number of minutes
        :param sec: number of seconds
        :return: frame index converted from time format
        :rtype: int
        """
        return (3600 * hour + minute * 60 + sec) * self.fps

    @property
    def step(self) -> int:
        """Get self._step

        :return: self._step
        :rtype: int
        """
        return self._step

    @step.setter
    def step(self, step: int) -> None:
        """Set self._step.

        :param step: step for the next index in comparing
        :return: set step
        :rtype: None
        """
        if step < 1:
            raise ValueError("Step should be positive integer")
        self._step = step

    @property
    def threshold(self) -> float:
        """Get self._threshold

        :return: self._threshold
        :rtype: float
        """
        return self._threshold

    @threshold.setter
    def threshold(self, threshold: float) -> None:
        """Set self._threshold.

        :param threshold: value to compare with error
        :return: set threshold
        :rtype: None
        """
        self._threshold = threshold
