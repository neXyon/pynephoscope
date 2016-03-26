#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2016 Joerg Hermann Mueller
#
# This file is part of pynephoscope.
#
# pynephoscope is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pynephoscope is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pynephoscope.  If not, see <http://www.gnu.org/licenses/>.

import sys
import cv2
import os.path
import pickle
from PyQt5 import QtCore, QtGui, QtWidgets
from main_ui import Ui_MainWindow
from image_view_ui import Ui_ImageWidget
from settings_view_ui import Ui_SettingsWidget
from configuration import Configuration
from skycamerafile import SkyCameraFile
from cloud_detection import *
from frame_difference import FrameDifference
from star_checker import StarCheckerHelper
from star_detection import *
from calibration import StarCorrespondence
from configuration import Configuration

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt


class PlotWidget(QtWidgets.QWidget):
	def __init__(self, parent=None):
		QtWidgets.QWidget.__init__(self, parent)
		self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
		
		self.figure = plt.figure()
		
		self.canvas = FigureCanvas(self.figure)
		
		self.toolbar = NavigationToolbar(self.canvas, self)
		
		self.button = QtWidgets.QPushButton("Plot")
		self.button.clicked.connect(self.plot)
		
		layout = QtWidgets.QVBoxLayout()
		layout.addWidget(self.toolbar)
		layout.addWidget(self.canvas)
		layout.addWidget(self.button)
		
		self.setLayout(layout)
	
	def plot(self):
		data = [x for x in range(10)]
		
		ax = self.figure.add_subplot(111)
		
		ax.hold(False)
		
		ax.plot(data, "*-")
		
		self.canvas.draw()


class MainQT5(QtWidgets.QMainWindow):
	def __init__(self, parent=None):
		QtWidgets.QWidget.__init__(self, parent)
		
		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)
		
		self.ui.action_New_View.triggered.connect(self.new_view)
		self.ui.action_Tile.triggered.connect(self.tile)
		self.ui.action_Settings.triggered.connect(self.settings)
		
		self.new_view().showMaximized()
		
		#self.plot()
		
		self.loadSettings()

	def plot(self):
		widget = PlotWidget()
		self.ui.mdiArea.addSubWindow(widget);
		widget.show()
		return widget
	
	def settings(self):
		widget = SettingsWidget()
		self.ui.mdiArea.addSubWindow(widget);
		widget.show()
		return widget
	
	def new_view(self):
		widget = ImageWidget()
		self.ui.mdiArea.addSubWindow(widget);
		widget.show()
		return widget
	
	def tile(self):
		if self.ui.mdiArea.currentSubWindow().isMaximized():
			self.ui.mdiArea.currentSubWindow().showNormal()
		
		position = QtCore.QPoint(0, 0)
		
		for window in self.ui.mdiArea.subWindowList():
			rect = QtCore.QRect(0, 0, self.ui.mdiArea.width(), self.ui.mdiArea.height() / len(self.ui.mdiArea.subWindowList()))
			window.setGeometry(rect)
			window.move(position)
			position.setY(position.y() + window.height())
	
	def loadSettings(self):
		if not os.path.exists(Configuration.configuration_file):
			return
		
		dictionary = None
		
		with open(Configuration.configuration_file, 'rb') as f:
			dictionary = pickle.load(f)
		
		for name, value in dictionary.items():
			setattr(Configuration, name, value)


class ImageViewMode():
	def __init__(self, view):
		self.view = view
	
	def getImage(self):
		return self.view.image

class CloudViewMode():
	def __init__(self, view):
		self.view = view
		self.detectors = []
		self.detectors.append(CDRBDifference())
		self.detectors.append(CDRBRatio())
		self.detectors.append(CDBRRatio())
		self.detectors.append(CDNBRRatio())
		self.detectors.append(CDAdaptive())
		self.detectors.append(CDMulticolor())
		self.detectors.append(CDBackground())
		self.detectors.append(CDSuperPixel())
		self.helper = CloudDetectionHelper()
		self.current_detector = 0
	
	def setDetector(self, index):
		self.current_detector = index
		self.view.refresh()
	
	def getImage(self):
		image = self.view.image
		result = self.helper.close_result(self.detectors[self.current_detector].detect(image, self.helper.get_mask(image)))
		return self.helper.get_result_image(result)

class StarViewMode():
	def __init__(self, view):
		self.view = view
		self.detectors = []
		self.detectors.append(GaussianStarFinder())
		self.detectors.append(CandidateStarFinder(FASTStarDetector()))
		self.detectors.append(CandidateStarFinder(GFTTStarDetector()))
		self.detectors.append(CandidateStarFinder(SURFStarDetector()))
		self.detectors.append(CandidateStarFinder(LoGStarDetector()))
		self.helper = StarCheckerHelper(Configuration.calibration_file)
		self.current_detector = 0
	
	def setDetector(self, index):
		self.current_detector = index
		self.view.refresh()
	
	def getImage(self):
		self.helper.prepare(self.view.files[self.view.index], self.detectors[self.current_detector])
		
		return self.helper.get_image()

class DifferenceViewMode():
	def __init__(self, view):
		self.difference = FrameDifference()
		self.view = view
		self.differences = np.array([])
		self.window_size = Configuration.difference_detection_window_size
	
	def reset(self):
		self.differences = np.zeros(len(self.view.files))
		self.differences[:] = np.nan
	
	def getImage(self):
		image1 = self.view.image
		image2 = image1
		if self.view.index > 0:
			image2 = cv2.imread(self.view.files[self.view.index - 1], 1)
		
		self.difference.difference(image1, image2)
		self.differences[self.view.index] = self.difference.getValue()
		return self.difference.getImage()
	
	def nextInteresting(self, backward = False):
		length = len(self.differences)
		
		if length < self.window_size:
			# err: not enough files
			if backward:
				self.view.selectFile(0)
			else:
				self.view.selectFile(len(self.differences) - 1)
			return
		
		step = 1
		if backward:
			step = -1
		
		index = self.view.index
		
		start = index - step * self.window_size
		if start < 0:
			start = 0
		if start >= length:
			start = length - 1
		
		end = length - 1
		if backward:
			end = 0
		
		for i in range(start, start + step * (self.window_size + 1), step):
			if np.isnan(self.differences[i]):
				self.view.selectFile(i)
				if self.view.modes[self.view.current_mode] != self.view.difference_mode:
					self.getImage()
		
		run = True
		
		index = start + step * self.window_size
		
		while run:
			index += step
			if index == end + step or index == end:
				index = end
				break
			
			start += step
			
			if np.isnan(self.differences[index]):
				self.view.selectFile(index)
				if self.view.modes[self.view.current_mode] != self.view.difference_mode:
					self.getImage()
			
			window = self.differences[start:start + step * self.window_size:step]
			mean = np.median(window)
			differences = np.abs(self.differences[start+step:start + step * (self.window_size + 1):step] - window)
			stdev = np.median(differences)
			
			if self.differences[index] > mean + stdev * 2:
				run = False
		
		self.view.selectFile(index)

class ImageWidget(QtWidgets.QWidget):
	def __init__(self, parent=None):
		QtWidgets.QWidget.__init__(self, parent)
		self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
		
		self.modes = []
		self.modes.append(ImageViewMode(self))
		self.modes.append(CloudViewMode(self))
		self.modes.append(StarViewMode(self))
		self.difference_mode = DifferenceViewMode(self)
		self.modes.append(self.difference_mode)
		
		self.current_mode = 0
		self.index = 0
		self.files = []
		
		self.ui = Ui_ImageWidget()
		self.ui.setupUi(self)
		
		self.filesystemmodel = QtWidgets.QFileSystemModel(self)
		self.filesystemmodel.setFilter(QtCore.QDir.NoDotAndDotDot | QtCore.QDir.AllDirs)
		self.filesystemmodel.setRootPath("/")
		self.ui.folderView.setModel(self.filesystemmodel)
		self.ui.folderView.hideColumn(1)
		self.ui.folderView.hideColumn(2)
		self.ui.folderView.hideColumn(3)
		self.ui.folderView.setCurrentIndex(self.filesystemmodel.index(Configuration.default_storage_path))
		self.ui.folderView.clicked.connect(self.changePath)
		self.changePath(self.ui.folderView.currentIndex())
		
		self.ui.imageSelector.valueChanged.connect(self.selectImage)
		self.ui.firstButton.clicked.connect(self.firstFile)
		self.ui.previousButton.clicked.connect(self.previousFile)
		self.ui.nextButton.clicked.connect(self.nextFile)
		self.ui.lastButton.clicked.connect(self.lastFile)
		
		self.ui.rbImage.clicked.connect(self.showImage)
		self.ui.rbClouds.clicked.connect(self.showClouds)
		self.ui.rbStars.clicked.connect(self.showStars)
		self.ui.rbDifference.clicked.connect(self.showDifference)
		
		self.ui.cbRefresh.toggled.connect(self.toggleAutoRefresh)
		
		self.ui.cbAlgorithm.currentIndexChanged.connect(self.modes[1].setDetector)
		self.ui.cbStars.currentIndexChanged.connect(self.modes[2].setDetector)
		
		self.ui.diffPrevious.clicked.connect(self.previousFile)
		self.ui.diffNext.clicked.connect(self.nextFile)
		self.ui.diffPreviousInteresting.clicked.connect(self.previousInteresting)
		self.ui.diffNextInteresting.clicked.connect(self.nextInteresting)
		
		self.timer = QtCore.QTimer(self)
		self.timer.setInterval(500)
		self.timer.timeout.connect(self.refresh)
	
	def toggleAutoRefresh(self, enable):
		if enable:
			self.timer.start()
		else:
			self.timer.stop()
	
	def previousInteresting(self):
		self.difference_mode.nextInteresting(True)
	
	def nextInteresting(self):
		self.difference_mode.nextInteresting(False)
	
	def activateMode(self, mode):
		if mode == self.current_mode:
			return
		
		self.current_mode = mode
		self.selectImage(self.index)
	
	def showImage(self):
		self.activateMode(0)
	
	def showClouds(self):
		self.activateMode(1)
	
	def showStars(self):
		self.activateMode(2)
	
	def showDifference(self):
		self.activateMode(3)
	
	def changePath(self, index):
		path = self.filesystemmodel.fileInfo(index).absoluteFilePath()
		self.files = SkyCameraFile.glob(path)
		if len(self.files) > 0:
			self.ui.imageSelector.setMaximum(len(self.files) - 1)
			self.ui.imageSelector.setEnabled(True)
		else:
			self.ui.imageSelector.setEnabled(False)
		
		self.difference_mode.reset()
		self.selectFile(0)
		
	def firstFile(self):
		self.selectFile(0)
		
	def previousFile(self):
		self.selectFile(self.ui.imageSelector.sliderPosition() - 1)
		
	def nextFile(self):
		self.selectFile(self.ui.imageSelector.sliderPosition() + 1)
		
	def lastFile(self):
		self.selectFile(len(self.files) - 1)
	
	def selectFile(self, index):
		self.ui.imageSelector.setSliderPosition(index)
		self.selectImage(self.ui.imageSelector.sliderPosition())
	
	def refresh(self):
		self.selectImage(self.index)
	
	def selectImage(self, index):
		if index >= len(self.files) or index < 0:
			self.ui.imageView.setText("No images found.")
			return
		
		self.index = index
		self.image = cv2.imread(self.files[index], 1)
		
		image = self.modes[self.current_mode].getImage()
		
		if len(image.shape) < 3 or image.shape[2] == 1:
			image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
		else:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		
		height, width, byteValue = self.image.shape
		byteValue = byteValue * width
		
		qimage = QtGui.QImage(image, width, height, byteValue, QtGui.QImage.Format_RGB888)
		
		self.ui.imageView.setPixmap(QtGui.QPixmap.fromImage(qimage))

class CSpinBox:
	def __init__(self, name, ui):
		self.name = name
		self.ui = getattr(ui, name)
		
		self.ui.setValue(getattr(Configuration, name))
		self.ui.valueChanged.connect(self.updateValue)
	
	def updateValue(self, value):
		setattr(Configuration, self.name, value)

class CCheckBox:
	def __init__(self, name, ui):
		self.name = name
		self.ui = getattr(ui, name)
		
		self.ui.setChecked(getattr(Configuration, name))
		self.ui.toggled.connect(self.updateValue)
	
	def updateValue(self, value):
		setattr(Configuration, self.name, value)

class CPathBox:
	def __init__(self, name, ui, button):
		self.name = name
		self.ui = getattr(ui, name)
		self.button = button
		
		self.ui.setText(getattr(Configuration, name))
		self.ui.textChanged.connect(self.updatePath)
		button.clicked.connect(self.selectPath)
	
	def selectPath(self):
		directory = QtWidgets.QFileDialog.getExistingDirectory(None, "Choose directory", getattr(Configuration, self.name))
		
		if os.path.exists(directory) and os.path.isdir(directory):
			setattr(Configuration, self.name, directory)
			self.ui.setText(directory)
	
	def updatePath(self, directory):
		if os.path.exists(directory) and os.path.isdir(directory):
			setattr(Configuration, self.name, directory)
	
class SettingsWidget(QtWidgets.QWidget):
	def __init__(self, parent=None):
		QtWidgets.QWidget.__init__(self, parent)
		self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
		
		self.ui = Ui_SettingsWidget()
		self.ui.setupUi(self)
		
		self.elements = []
		
		self.elements.append(CSpinBox("day_averaging_frames", self.ui))
		self.elements.append(CSpinBox("night_averaging_frames", self.ui))
		self.elements.append(CSpinBox("day_time_between_frames", self.ui))
		self.elements.append(CSpinBox("night_time_between_frames", self.ui))
		self.elements.append(CPathBox("default_storage_path", self.ui, self.ui.btStorage))
		self.elements.append(CCheckBox("store_in_subdirectory", self.ui))
		self.elements.append(CCheckBox("dnfa_enabled", self.ui))
		self.elements.append(CSpinBox("dnfa_window_size", self.ui))
		self.elements.append(CSpinBox("dnfa_min_med_diff_factor", self.ui))
		self.elements.append(CSpinBox("dnfa_min_diff_value", self.ui))
		self.elements.append(CSpinBox("dnfa_min_frames", self.ui))
		self.elements.append(CSpinBox("dnfa_max_frames", self.ui))
		
		self.elements.append(CSpinBox("rb_difference_threshold", self.ui))
		self.elements.append(CSpinBox("rb_ratio_threshold", self.ui))
		self.elements.append(CSpinBox("br_ratio_threshold", self.ui))
		self.elements.append(CSpinBox("nbr_threshold", self.ui))
		self.elements.append(CSpinBox("adaptive_threshold", self.ui))
		self.elements.append(CSpinBox("adaptive_block_size", self.ui))
		self.elements.append(CSpinBox("background_rect_size", self.ui))
		self.elements.append(CSpinBox("mc_rb_threshold", self.ui))
		self.elements.append(CSpinBox("mc_bg_threshold", self.ui))
		self.elements.append(CSpinBox("mc_b_threshold", self.ui))
		self.elements.append(CSpinBox("sp_num_superpixels", self.ui))
		self.elements.append(CSpinBox("sp_prior", self.ui))
		self.elements.append(CSpinBox("sp_num_levels", self.ui))
		self.elements.append(CSpinBox("sp_num_histogram_bins", self.ui))
		self.elements.append(CSpinBox("sp_num_iterations", self.ui))
		self.elements.append(CSpinBox("sp_kernel_size", self.ui))
		self.elements.append(CSpinBox("morphology_kernel_size", self.ui))
		self.elements.append(CSpinBox("morphology_iterations", self.ui))
		
		self.elements.append(CSpinBox("min_alt", self.ui))
		self.elements.append(CSpinBox("alt_step", self.ui))
		self.elements.append(CSpinBox("az_step", self.ui))
		self.elements.append(CSpinBox("max_mag", self.ui))
		self.elements.append(CSpinBox("gaussian_roi_size", self.ui))
		self.elements.append(CSpinBox("gaussian_threshold", self.ui))
		self.elements.append(CSpinBox("gaussian_kernel_size", self.ui))
		self.elements.append(CSpinBox("candidate_radius", self.ui))
		self.elements.append(CSpinBox("gftt_max_corners", self.ui))
		self.elements.append(CSpinBox("gftt_quality_level", self.ui))
		self.elements.append(CSpinBox("gftt_min_distance", self.ui))
		self.elements.append(CSpinBox("surf_threshold", self.ui))
		self.elements.append(CSpinBox("log_max_rect_size", self.ui))
		self.elements.append(CSpinBox("log_block_size", self.ui))
		self.elements.append(CSpinBox("log_threshold", self.ui))
		self.elements.append(CSpinBox("log_kernel_size", self.ui))
	
	def saveSettings(self):
		dictionary = {}
		for element in self.elements:
			name = element.name
			dictionary[name] = getattr(Configuration, name)
			
		with open(Configuration.configuration_file, 'wb') as f:
			pickle.dump(dictionary, f)
	
	def closeEvent(self, event):
		button = QtWidgets.QMessageBox.question(self, "Closing settings", "Save changes?", QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel)
		
		if button == QtWidgets.QMessageBox.Save:
			self.saveSettings()
		elif button == QtWidgets.QMessageBox.Cancel:
			event.ignore()
			return
		
		event.accept()

if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	main_window = MainQT5()
	main_window.show()
	sys.exit(app.exec_())

