import cv2
import numpy as np

from logger.logger import logger
"""
This class reads frames from the camera 
Splits the frames into their respective RGB channels.
Displays the frames in a single window with two on top and two on the bottom.
"""
class SplitChannelRGB:
    def __init__(self, camera_number: int = 0):
        self.camera_number = camera_number
        self.cap = cv2.VideoCapture(self.camera_number)

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            logger.error("Error reading frame.")
        return frame

    def show_frame(self, frame):
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            logger.info("Frames displayed.")

    # get r, g , b channels and original
    def get_channels(self, frame):
        b, g, r = cv2.split(frame)
        zeros = np.zeros(frame.shape[:2], dtype="uint8")
        b_channel = cv2.merge([b, zeros, zeros])
        g_channel = cv2.merge([zeros, g, zeros])
        r_channel = cv2.merge([zeros, zeros, r])
        return b_channel, g_channel, r_channel

    def stack_frames(self, b_channel, g_channel, r_channel, frame):
        top_frames = np.hstack((b_channel, g_channel))
        bottom_frames = np.hstack((r_channel, frame))
        all_frames = np.vstack((top_frames, bottom_frames))
        return all_frames

    def run(self):
        while True:
            frame = self.read_frame()
            # self.show_frame(frame)
            b_channel, g_channel, r_channel = self.get_channels(frame)
            # display the frames in a single window with two on top and two bottom
            all_frames = self.stack_frames(b_channel, g_channel, r_channel, frame)
            cv2.imshow("Frames", all_frames)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Frames displayed.")

    def print(self):
        logger.info(f"Camera number: {self.camera_number}")

if __name__ == "__main__":
    rgb_frames = SplitChannelRGB()
    rgb_frames.print()
    rgb_frames.run()