import sys
from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QDialog, QFileDialog
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread
import cv2
import time
import mediapipe as mp
import os
import pyautogui as pg

from PyQt6.QtCore import pyqtSignal, QObject

class Signals(QObject):
    image_display = pyqtSignal(object)

    loading_progress_signal_started = pyqtSignal()
    loading_progress_signal_working = pyqtSignal(int)
    loading_progress_signal_finished = pyqtSignal()

    show_status = pyqtSignal(str)

    camera_started = pyqtSignal()
    camera_stoped = pyqtSignal()

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
    # TODO : use this to implement the actions radioboxs
    cam: cv2.VideoCapture

    selected_action = 0

    def change_mode(self, value):
        self.selected_action = value

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

        self.wCam, self.hCam = 640, 480

    def set_camera(self, cam: cv2.VideoCapture):
        self.cam = cam

    def run(self):
        self.signals.camera_started.emit()
        self.is_stopped = False
        while not self.is_stopped & self.cam.isOpened():
            try:
                with self.mp_hands.Hands(model_complexity=self.model_complexity, min_detection_confidence=self.min_detection_confidence, min_tracking_confidence=self.min_tracking_confidence) as hands:
                    success, image = self.cam.read()

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = hands.process(image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    if not success:
                        continue
                    
                    
                    if self.selected_action == 2:
                        self.mouse_controle(results)

                    elif self.selected_action == 1:
                        self.finger_counter(image, results)

                    else:
                        self.sample_aquisition(image)


                    self.signals.image_display.emit(image)

            except Exception as ex:
                pass   
                
        self.signals.camera_stoped.emit()
        self.cam.release()

    def stop(self):
        self.is_stopped = True
        
    def sample_aquisition(self, image):
        return image

    def mouse_controle(self, results):
        sensitivity = 2
        for hand_landmarks in results.multi_hand_landmarks:
                handLandmarks = []
                for landmark in hand_landmarks.landmark:
                    handLandmarks.append([landmark.x, landmark.y])
        x1, y1 = int(handLandmarks[4][0] * self.wCam), int(handLandmarks[4][1] * self.hCam)  # Thumb finger
        pg.moveTo(x1 * sensitivity, y1 * sensitivity)

    def finger_counter(self, image, results):
        fingerCount = 0 
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                handLabel = results.multi_handedness[handIndex].classification[0].label
                handLandmarks = []
                for landmark in hand_landmarks.landmark:
                    handLandmarks.append([landmark.x, landmark.y])
  
                if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
                    fingerCount = fingerCount+1
                    cv2.putText(image,"Right raised", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
                    fingerCount = fingerCount+1
                    cv2.putText(image,"Left raised", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    
                if handLandmarks[8][1] < handLandmarks[6][1]:#Index finger
                    fingerCount = fingerCount+1
                if handLandmarks[12][1] < handLandmarks[10][1]:#Middle finger
                    fingerCount = fingerCount+1
                if handLandmarks[16][1] < handLandmarks[14][1]:#Ring finger
                    fingerCount = fingerCount+1
                if handLandmarks[20][1] < handLandmarks[18][1]:#little finger
                    fingerCount = fingerCount+1

                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
                
            cv2.putText(image, f'Finger up={str(fingerCount)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return image 
    
class MainWindow(QMainWindow):

    camera : object = None
    bgCamera : CameraWorker = None
    default_image = cv2.imread("resources/images/blackscreen.png")
    saveIndex = 1
    output_folder = "images/"
    mode = 0

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
        
        self.display_image(self.default_image)
    
    def check_available_cameras(self):
        available_cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(cap)
                cap.release()
        return available_cameras
    
    def select_action(self, mode):
        self.mode = mode

    def connect_events(self):

        self.rbNONE.toggled.connect(lambda : self.select_action(0))
        self.rbCD.toggled.connect(lambda : self.select_action(1))
        self.rbCS.toggled.connect(lambda : self.select_action(2))

        events_actions = {
            self.signals.image_display : self.display_image,
            self.cbCamera.currentIndexChanged : self.camera_changed,
            self.signals.show_status : lambda msg : self.statusbar.showMessage(msg),
            self.signals.camera_started : self.camera_started,
            self.signals.camera_stoped : self.camera_stoped,
        }

        for event, action in events_actions.items():
            event.connect(action)

        buttons_actions = {
            self.btnStart : self.btnStart_action,
            self.btnStop : self.btnStop_action,
            self.btnSave : self.btnSave_action,
            self.btnSavePath : self.btnSave_path,
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
        self.bgCamera.change_mode(self.mode)
        self.bgCamera.start()

    def btnStop_action(self):
        if self.bgCamera:
            self.bgCamera.stop()
            self.bgCamera = None
            self.display_image(self.default_image)

    def btnSave_action(self):
        images_filename = os.path.join(self.output_folder, f'image_{self.saveIndex}.png')
        cv2.imwrite(images_filename, self.current_image)
        self.saveIndex += 1
        self.signals.show_status.emit(f"Image enregistrée avec succès dans : {self.output_folder}")

    def btnSave_path(self):       
        options = QFileDialog.Option
        options = QFileDialog.Option.ShowDirsOnly 
        options |= QFileDialog.Option.DontUseNativeDialog

        directory = QFileDialog.getExistingDirectory(self, "Choisir un répertoire d'enrengistrement", self.output_folder, options=options)
        if directory:
            self.output_folder = directory
            self.signals.show_status.emit(f"Répertoire d'enregistrement sélectionné : {directory}")
        else:
            self.signals.show_status.emit("Répertoire d'enregistrement non sélectionné.")



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

    def display_image(self, image):

        self.current_image = image

        if self.rbNegatif.isChecked():
            image = abs(255 - image)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)

        elif self.rbGRAY.isChecked():
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = image.shape
            #help
            q_image = QImage(image.data, width, height, width, QImage.Format.Format_Grayscale8)
        else:
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)


        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

        

    def camera_started(self):
        self.btnStart.setEnabled(False)
        self.box_action.setEnabled(False)
        self.cbCamera.setEnabled(False)

        self.btnSave.setEnabled(True)
        self.btnStop.setEnabled(True)

        self.display_image(self.default_image)

    def camera_stoped(self):
        self.btnStart.setEnabled(True)
        self.box_action.setEnabled(True)
        self.cbCamera.setEnabled(True)

        self.btnSave.setEnabled(False)
        self.btnStop.setEnabled(False)

        self.display_image(self.default_image)

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
        self.progressFrame.setFixedWidth(630)

        self.main = MainWindow()
        self.main.show()

        self.close()
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    screen = Splash()
    screen.show()
    sys.exit(app.exec())