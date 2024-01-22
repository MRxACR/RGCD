import sys
from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QDialog
from PyQt6.QtGui import QPixmap, QImage, QImageReader
from PyQt6.QtCore import Qt, QThread
import cv2
import time
import mediapipe as mp

from PyQt6.QtCore import pyqtSignal, QObject

class Signals(QObject):
    image_display = pyqtSignal(object)

    loading_progress_signal_started = pyqtSignal()
    loading_progress_signal_working = pyqtSignal(int)
    loading_progress_signal_finished = pyqtSignal()

    show_status = pyqtSignal(str)

class LoadingWorker(QThread):
    """
    This class is used when the app starts `Splash screen`, you can used it to implement `loading` logic,
    it displays the `splash screen` loading.
    """
    def __init__(self, signals : Signals):
        super().__init__()
        self.signals = signals

    def run(self):
        self.signals.loading_progress_signal_started.emit()
        for i in range(101):
            break
            self.signals.loading_progress_signal_working.emit(i)
            time.sleep(0.02)
        self.signals.loading_progress_signal_finished.emit()

class CameraWorker(QThread):
    
    cam: cv2.VideoCapture

    def __init__(self, signals : Signals):
        super().__init__()
        self.signals = signals

        self.model_complexity=0
        self.min_detection_confidence=0.5
        self.min_tracking_confidence=0.5

        # mediapipe initialize
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

    def set_camera(self, cam: cv2.VideoCapture):
        self.cam = cam

    def run(self):
        self.signals.show_status.emit("Camera a démarer")
        self.is_stopped = False
        while not self.is_stopped & self.cam.isOpened():
            try:
                with self.mp_hands.Hands(model_complexity=self.model_complexity, min_detection_confidence=self.min_detection_confidence, min_tracking_confidence=self.min_tracking_confidence) as hands:
                    success, image = self.cam.read()
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = hands.process(image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(
                                image,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                self.mp_drawing_styles.get_default_hand_connections_style()
                                )
                            
                    # find postion of Hand landmarks      
                    lmList = []
                    if results.multi_hand_landmarks:
                        myHand = results.multi_hand_landmarks[0]
                        for id, lm in enumerate(myHand.landmark):
                            h, w, c = image.shape
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            lmList.append([id, cx, cy])          

                    # Assigning variables for fingers position
                    if len(lmList) > 0:
                        x1, y1 = lmList[4][1], lmList[4][2] # Thumb finger
                        x2, y2 = lmList[8][1], lmList[8][2] #Index finger
                        x3, y3 = lmList[12][1], lmList[12][2]  # Middle finger
                        x4, y4 = lmList[16][1], lmList[16][2]  # Ring finger
                        x5, y5 = lmList[20][1], lmList[20][2]  # Little finger

                    self.signals.image_display.emit(image)
            except Exception as ex:
                pass   
                
        self.signals.show_status.emit("Caméra est arrêtée")
        self.cam.release()

    def stop(self):
        self.is_stopped = True
        
class MainWindow(QMainWindow):

    camera : object = None
    bgCamera : CameraWorker = None

    def __init__(self):
        super().__init__()
        uic.loadUi("resources/gui/home.ui", self)

        self.signals = Signals()

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.frame_layout = QVBoxLayout(self.frCamera)

        self.image_label = QLabel(self.frCamera)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_layout.addWidget(self.image_label)

        self.setup()
        self.connect_events()

    def setup(self):
        self.available_cameras = self.check_available_cameras()

        for i, cap in enumerate(self.available_cameras):
            camera_info = {
                'index': i,
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'brightness': cap.get(cv2.CAP_PROP_BRIGHTNESS),
                'contrast': cap.get(cv2.CAP_PROP_CONTRAST),
                'saturation': cap.get(cv2.CAP_PROP_SATURATION),
                'hue': cap.get(cv2.CAP_PROP_HUE),
                'gain': cap.get(cv2.CAP_PROP_GAIN),
                'exposure': cap.get(cv2.CAP_PROP_EXPOSURE),
                'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC)),
                'is_color': cap.get(cv2.CAP_PROP_CONVERT_RGB),
            }
            camera_label = f"Camera {camera_info['index']} - {camera_info['width']}x{camera_info['height']} - {camera_info['fps']:.2f} fps"
            self.cbCamera.addItem(camera_label)
    
    def check_available_cameras(self):
        available_cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(cap)
                cap.release()
        return available_cameras

    def connect_events(self):

        events_actions = {
            self.signals.image_display : self.display_image,
            self.cbCamera.currentIndexChanged : self.camera_changed,
            self.signals.show_status : lambda msg : self.statusbar.showMessage(msg),
        }

        for event, action in events_actions.items():
            event.connect(action)

        buttons_actions = {
            self.btnStart : self.btnStart_action,
            self.btnStop : self.btnStop_action,
            self.btnSave : self.btnSave_action,
        }

        for buttons, action in buttons_actions.items():
            buttons.clicked.connect(action)

    def btnStart_action(self):
        if not self.camera:
            self.signals.show_status.emit("Erreur : Aucune camera n'a été sélectionné")
            return 
        self.camera_changed()
        self.bgCamera = CameraWorker(self.signals)
        self.bgCamera.set_camera(self.camera)
        self.bgCamera.start()

    def btnStop_action(self):
        if self.bgCamera:
            self.bgCamera.stop()
            self.bgCamera = None

    def btnSave_action(self):
        pass

    def camera_changed(self):

        original_string = self.cbCamera.currentText()

        if len(self.available_cameras) == 0 or original_string == "Aucune" : 
            self.camera = None
            self.signals.show_status.emit(f"Aucune camera est sélectionnée")
            return

        begin_text = "Camera "
        end_text = " - "

        # Find the starting index of the begin_text
        begin_index = original_string.find(begin_text)

        # Find the ending index of the end_text
        end_index = original_string.find(end_text, begin_index + len(begin_text))

        # Extract the index
        index = int(original_string[begin_index + 7:end_index])

        self.camera = cv2.VideoCapture(index)

        self.signals.show_status.emit(f"Camera {index} est sélectionnée")   

    def display_image(self, image : str):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

class Splash(QDialog):
    def __init__(self,parent=None):
        super().__init__(parent)
        uic.loadUi("resources/gui/splash.ui", self)
        
        self.signals = Signals()
        self.bw = LoadingWorker(self.signals)
        self.connect_events()
        self.setup()

    def setup(self):
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.bw.start()

    def connect_events(self):
        self.signals.loading_progress_signal_started.connect( self.loading_started  )
        self.signals.loading_progress_signal_working.connect( self.loading_progress  )
        self.signals.loading_progress_signal_finished.connect(  self.loading_finish )
    
    def loading_started(self ):
        self.progressFrame.setFixedWidth(0)

    def loading_progress(self, i: float ):
        j = int ( i * 630 / 100)
        self.progressFrame.setFixedWidth(j)

    def loading_finish(self):
        self.progressFrame.setFixedWidth(430)

        self.main = MainWindow()
        self.main.show()

        self.close()
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    screen = Splash()
    screen.show()
    sys.exit(app.exec())