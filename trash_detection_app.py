import os
#Use cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#use gpu
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PyQt5.QtCore import * 
from PyQt5.QtGui import * 
from PyQt5.QtWidgets import *
import os
import pathlib
import json
import PIL
import cv2
import numpy as np
import tensorflow as tf 
from PIL import Image
from google.protobuf import text_format
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_utils
from object_detection.protos import pipeline_pb2
from object_detection.builders import model_builder
import IPython.display as display
import sys
import time

SSD640_CHECKPOINT_NUM = 151
SSD640_MODEL_DIR = "./my_models/640x640ssd/saved"
SSD640_CONFIG_FILE = "./my_models/640x640ssd/pipeline.config"
SSD_FRAME_SIZE = 640
CN_CHECKPOINT_NUM = 48
CN_MODEL_DIR = "./my_models/cn"
CN_CONFIG_FILE = CN_MODEL_DIR + "/pipeline.config"
CN_FRAME_SIZE = 512

CNMN_CHECKPOINT_NUM = 138
CNMN_MODEL_DIR = "./my_models/cnmn"
CNMN_CONFIG_FILE = CNMN_MODEL_DIR + "/pipeline.config"
CNMN_FRAME_SIZE = 512

SCORE_THRESHOLD = 0.5
NUM_BOXES = 10

class Detector(QWidget):
	class Worker(QObject):
		finished = pyqtSignal()

		def __init__(self, detector):
			super().__init__()
			self.detector = detector

		#This function contains the while loop that continuously 
		#takes images from a webcam and feeds them to the model
		def run_detection_loop(self):
			label_map_file = "C:/Users/Randon/Desktop/Projects/ai-trash-detector/data/TACO-master/data/label_map.pbtxt"
			cat_index = label_map_util.create_category_index_from_labelmap(label_map_file)

			cap = cv2.VideoCapture(0)
			width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			print("width: {}, height: {}".format(width, height))
			
			if(self.detector.current_model == 0):
				self.cur_model = self.detector.ssd_model
				frame_size = SSD_FRAME_SIZE
			elif(self.detector.current_model == 1):
				self.cur_model = self.detector.ed_model
				frame_size = CNMN_FRAME_SIZE
			else:
				self.cur_model = self.detector.cn_model
				frame_size = CN_FRAME_SIZE

			while True:
				start = time.time()
				ret, frame = cap.read()
				image_np = cv2.resize(np.array(frame), (frame_size, frame_size))
				#image_np = np.array(frame)
				input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
				detections = self.detect_func(input_tensor)

				num_detections = int(detections.pop('num_detections'))
				detections = {key:value[0, :num_detections].numpy() for key, value in detections.items()}
				detections['num_detections'] = num_detections
				detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

				label_id_offset = 1
				image_np_with_detections = image_np.copy()

				vis_utils.visualize_boxes_and_labels_on_image_array(
							image_np_with_detections,
							detections['detection_boxes'],
							detections['detection_classes']+label_id_offset,
							detections['detection_scores'],
							cat_index,
							use_normalized_coordinates=True,
							max_boxes_to_draw=NUM_BOXES,
							min_score_thresh=SCORE_THRESHOLD,
							agnostic_mode=False)

				self.disp_image = cv2.resize(image_np_with_detections, (640, 480))
				self.disp_image = QImage(self.disp_image.data, self.disp_image.shape[1], self.disp_image.shape[0], QImage.Format_RGB888).rgbSwapped() 
				self.detector.image_frame.setPixmap(QPixmap.fromImage(self.disp_image))

				end = time.time()
				print(end - start)

				if self.detector.detection_running == False:
					cap.release()
					break

			cap.release()
			self.detector.image_frame.clear()
			self.finished.emit()
		
		#This function takes a selected image and feeds it to the model
		def image_detection(self):
			label_map_file = "C:/Users/Randon/Desktop/Projects/ai-trash-detector/data/TACO-master/data/label_map.pbtxt"
			cat_index = label_map_util.create_category_index_from_labelmap(label_map_file)
			
			#Set the current model based on what is selected in the application 
			if(self.detector.current_model == 0):
				self.cur_model = self.detector.ssd_model
				frame_size = SSD_FRAME_SIZE
			elif(self.detector.current_model == 1):
				self.cur_model = self.detector.ed_model
				frame_size = CNMN_FRAME_SIZE
			else:
				self.cur_model = self.detector.cn_model
				frame_size = CN_FRAME_SIZE

			start = time.time()
			self.image_data = cv2.resize(self.detector.image_data, (frame_size, frame_size))
			input_tensor = tf.convert_to_tensor(np.expand_dims(self.image_data, 0), dtype=tf.float32)
			detections = self.detect_func(input_tensor)

			num_detections = int(detections.pop('num_detections'))
			detections = {key:value[0, :num_detections].numpy() for key, value in detections.items()}
			detections['num_detections'] = num_detections
			detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

			label_id_offset = 1
			image_np_with_detections = self.image_data.copy()

			vis_utils.visualize_boxes_and_labels_on_image_array(
						image_np_with_detections,
						detections['detection_boxes'],
						detections['detection_classes']+label_id_offset,
						detections['detection_scores'],
						cat_index,
						use_normalized_coordinates=True,
						max_boxes_to_draw=NUM_BOXES,
						min_score_thresh=SCORE_THRESHOLD,
						agnostic_mode=False)
			end = time.time()
			print(end - start)
			self.disp_image = cv2.resize(image_np_with_detections, (640, 640))
			self.disp_image = QImage(self.disp_image.data, self.disp_image.shape[1], self.disp_image.shape[0], QImage.Format_RGB888).rgbSwapped() 
			self.detector.image_frame.setPixmap(QPixmap.fromImage(self.disp_image))

			print("Finished Detection")
			self.detector.stop_detection()
			self.finished.emit()

		#Takes an image and pases it to the selected model
		#Returns the detections
		def detect_func(self, image):
			image, shapes = self.cur_model.preprocess(image)
			prediction_dict = self.cur_model.predict(image, shapes)
			detections = self.cur_model.postprocess(prediction_dict,shapes)
			detections
			return detections

	def __init__(self):
		super().__init__()
		self.window_setup()
		self.import_models()
		self.detection_running = False
		self.mode = 0
		self.show()

	def window_setup(self):
		#Options layout
		self.options_layout = QVBoxLayout()

		#Radio buttons
		self.rb_group = QButtonGroup()

		self.rb1 = QRadioButton("Video")
		self.rb1.setChecked(True)
		self.rb2 = QRadioButton("Images")
		self.rb2.setChecked(False)
		self.rb1.toggled.connect(lambda:self.toggle_mode(self.rb1))

		self.rb_group.addButton(self.rb1)
		self.rb_group.addButton(self.rb2)

		self.options_layout.addWidget(self.rb1)
		self.options_layout.addWidget(self.rb2)

		self.mode = 0

		#Dropdown
		self.cb = QComboBox()
		self.cb.setEditable(True)
		self.cb.addItems(["MobileNet","CenterNet MobileNet","CenterNet"])
		self.current_model = 0
		self.cb.currentIndexChanged.connect(self.selectionchange)
		self.cb.lineEdit().setAlignment(Qt.AlignCenter)
		self.cb.lineEdit().setReadOnly(True)
		self.options_layout.addWidget(self.cb)

		#Button layout
		self.start_button = QPushButton('Start detection')
		self.stop_button = QPushButton('Stop detection')
		self.select_image_button = QPushButton('Select Image')
		self.start_button.clicked.connect(self.start_detection)
		self.stop_button.clicked.connect(self.stop_detection)
		self.select_image_button.clicked.connect(self.select_image)
		self.options_layout.addWidget(self.start_button)
		self.options_layout.addWidget(self.stop_button)
		self.options_layout.addWidget(self.select_image_button)

		#Image window layout
		self.image_frame = QLabel()
		self.image_frame.setFixedSize(640, 640)
		self.image_frame.setStyleSheet("border: 1px solid black;")
		self.image_frame_layout = QHBoxLayout()
		self.image_frame_layout.addWidget(self.image_frame, alignment=Qt.AlignCenter)

		self.image_frame_layout.addLayout(self.options_layout)
		self.setLayout(self.image_frame_layout)
	
	def toggle_mode(self, rb):
		if (self.detection_running == False):
			if (rb.isChecked() == True):
				self.mode = 0
			else:
				self.mode = 1
						
	def import_models(self):
		#Import ssd model
		ssd_config = config_util.get_configs_from_pipeline_file(SSD640_CONFIG_FILE)
		self.ssd_model = model_builder.build(model_config=ssd_config['model'], is_training=False)
		ssd_model_checkpoint = SSD640_MODEL_DIR + "/ckpt-{}".format(SSD640_CHECKPOINT_NUM)

		ssd_checkpoint = tf.compat.v2.train.Checkpoint(model=self.ssd_model)
		ssd_checkpoint.restore(ssd_model_checkpoint).expect_partial()

		#Import cn model
		cn_config = config_util.get_configs_from_pipeline_file(CN_CONFIG_FILE)
		self.cn_model = model_builder.build(model_config=cn_config['model'], is_training=False)
		cn_model_checkpoint = CN_MODEL_DIR + "/ckpt-{}".format(CN_CHECKPOINT_NUM)

		cn_checkpoint = tf.compat.v2.train.Checkpoint(model=self.cn_model)
		cn_checkpoint.restore(cn_model_checkpoint).expect_partial()

		#Import cnmn model
		ed_config = config_util.get_configs_from_pipeline_file(CNMN_CONFIG_FILE)
		self.ed_model = model_builder.build(model_config=ed_config['model'], is_training=False)
		ed_model_checkpoint = CNMN_MODEL_DIR + "/ckpt-{}".format(CNMN_CHECKPOINT_NUM)

		ed_checkpoint = tf.compat.v2.train.Checkpoint(model=self.ed_model)
		ed_checkpoint.restore(ed_model_checkpoint).expect_partial()

	def start_detection(self):
		if (self.detection_running == False):
			self.detection_running = True
			self.thread = QThread()
			self.worker = self.Worker(self)
			self.worker.moveToThread(self.thread)
			if (self.mode == 0):
				self.thread.started.connect(self.worker.run_detection_loop)
			else:
				self.thread.started.connect(self.worker.image_detection)
			
			self.worker.finished.connect(self.thread.quit)
			self.thread.start()

	def selectionchange(self,i):
		self.current_model = i

	def stop_detection(self):
		self.detection_running = False

	def select_image(self):
		if(self.mode == 1):
			fname = QFileDialog.getOpenFileName(self, 'Open file','C:/Users/Randon/Desktop/Projects/ai-trash-detector/data/TACO-master/data',"Image files (*.jpg)")
			if fname[0] != "":
				self.image_data = cv2.resize(cv2.imread(fname[0]), (640, 480))
				self.image = QImage(self.image_data.data, self.image_data.shape[1], self.image_data.shape[0], QImage.Format_RGB888).rgbSwapped() 
				self.image_frame.setPixmap(QPixmap(self.image))
				#self.image_frame.setPixmap(QPixmap.fromImage(fname))

def main():
	#Create window
	App = QApplication([])
	window = QWidget()
	window.resize(800, 600)
	window.setWindowTitle('Trash detector')

	#Create detector widget
	detector = Detector()
	layout = QStackedLayout()
	layout.addWidget(detector)

	#Add detector to the window
	window.setLayout(layout)

	window.show()

	#Exit when the window is closed
	sys.exit(App.exec_())

if __name__ == '__main__':
    main()