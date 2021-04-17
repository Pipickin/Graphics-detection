import numpy as np

import VideoComp as vc
import cv2 as cv

video_path = '/home/vlad/Videos/Barc_45760_49780.mp4'
model_path = 'save_modelV2'
barc_10_200 = vc.VideoComp(video_path, model_path, 1, 2)

steps = [1, 2, 3, 4]
source = [2075, 2161]
source2 = [x + 1 for x in source]
last = source + source2
print(last)
for index in last:
    barc_10_200.display_frame_by_index(index, wait=False, dynamic=True)

# for step in steps:
#     print(step)
#     barc_10_200.step = step
#     barc_10_200.compare_cap()
#
# cv.waitKey(0)
# cv.destroyAllWindows()


