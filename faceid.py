# import kivy dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.animation import Animation
import FaceDetectionModule as fd
# import other dependencies
import cv2
import tensorflow as tf
import os
import numpy as np
from layers import L1Dist

class CamApp(App):
    def build(self):
        # Main layout components 
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(
            text="Verify",
            on_press=self.verify,
            size_hint=(1, .1),
            background_color=(1, 0, 0, 1)  # Initial red
        )
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1, .1))
        
        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)
        
        # Load tensorflow/keras model
        self.model = tf.keras.models.load_model('mymodel.h5', custom_objects={'L1Dist': L1Dist})

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)
        
        return layout

    # Run continuously to get webcam feed
    def update(self, *args):
        ret, frame = self.capture.read()
        detector =fd.FaceDetector()
        frame = frame[120:120+250, 200:200+250,:]
        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture
        frame = detector.detectface(frame)

        # Flip and convert image to texture

    # Load and preprocess image
    def preprocess(self, file_path):
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)
        img = tf.image.resize(img, (100, 100))
        img = img / 255.0
        return img

    # Verification function
    def verify(self, *args):
        detection_threshold = 0.5
        verification_threshold = 0.5

        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)

        # Run model predictions
        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(SAVE_PATH)
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))
            result = self.model.predict([np.expand_dims(input_img, axis=0), np.expand_dims(validation_img, axis=0)])
            results.append(result)

        # Evaluate results
        detection = np.sum(np.array(results) > detection_threshold)
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
        verified = verification > verification_threshold

        self.verification_label.text = 'Verified' if verified else 'Unverified'

        # Animate color on success
        if verified:
        # Smoothly transition to blue in 0.5 seconds
         anim_to_blue = Animation(background_color=(0, 0, 1, 1), duration=2.5)
         anim_to_blue.start(self.button)
    
        # After 2.5 seconds, smoothly fade back to red in 1 second
         def revert_color(*_):
            anim_back = Animation(background_color=(1, 0, 0, 1), duration=0.5)
            anim_back.start(self.button)

         Clock.schedule_once(revert_color, 1)
        else:
            self.button.background_color = (1, 0, 0, 1)  # Immediately red on failure


        Logger.info(f"Results: {results}")
        Logger.info(f"Detection: {detection}")
        Logger.info(f"Verification: {verification}")
        Logger.info(f"Verified: {verified}")

        return results, verified

if __name__ == '__main__':
    CamApp().run()
