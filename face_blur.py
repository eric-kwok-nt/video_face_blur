import cv2
from moviepy.editor import *
import numpy as np
from os.path import dirname, join
import pdb


class FaceBlur:

    def __init__(self):
        self.prototxt_path = "./model/deploy.prototxt"
        self.model_path = "./model/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        self.video_path = "./Dataset/Video.mp4"

        #Load Caffe model
        self.model = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.model_path)

        self.threshold = 0.4
        # Confidence threshold

        self.pixelate_size = 8

    def process_images(self, image):
        h, w = image.shape[:2]

        # kernel_width = (w // 7) | 1
        # kernel_height = (h // 7) | 1
        # Gaussian Blur kernel size
        blob = cv2.dnn.blobFromImage(image, 1.0, None, (104.0, 177.0, 123.0))
        # Preprocess the image
        self.model.setInput(blob)
        # Set the image into the input of the neural network
        output = np.squeeze(self.model.forward())

        image_copy = image.copy()
        for i in range(0, output.shape[0]):
            confidence = output[i, 2]

            # Get confidence value
            if confidence > self.threshold:
                # print(confidence)
                box = output[i, 3:7] * np.array([w, h, w, h])
                # print(box)
                # Upscale the box to original image
                start_x, start_y, end_x, end_y = box.astype(int)
                # convert to integers
                face = image_copy[start_y: end_y, start_x: end_x]
                # Get the face image
                # face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 2)
                face_h, face_w = face.shape[:2]
                if face_h == 0 or face_w == 0:
                    continue
                # Resize input to "pixelated" size
                # pdb.set_trace()
                temp = cv2.resize(face, (self.pixelate_size, self.pixelate_size), interpolation=cv2.INTER_LINEAR)
                face = cv2.resize(temp, (face_w, face_h), interpolation=cv2.INTER_NEAREST)
                image_copy[start_y: end_y, start_x: end_x] = face
                # cv2.rectangle(image_copy,(start_x, start_y), (end_x, end_y), [0,255,0])

        return image_copy

    def edit_video(self):
        orig_clip = VideoFileClip(self.video_path)
        modified_clip = orig_clip.fl_image(self.process_images)
        modified_clip.write_videofile("Video_edited.mp4")



if __name__ == '__main__':
    FB = FaceBlur()

    # clip = VideoFileClip(FB.video_path)

    # frames = int(clip.fps * clip.duration)
    # for i in range(frames):
    #     image = FB.process_images(clip.get_frame(i))
    #     # image = clip.get_frame(1)
    #     # print(image)
    #     cv2.imshow('image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    #     cv2.waitKey(0)
    FB.edit_video()





