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

from astropy.time import Time
import os
import cv2
import numpy as np
import sys
from configuration import Configuration
from night import NightChecker
from control import SkyCameraControl
from skycamerafile import SkyCameraFile

class DynamicDifferenceThreshold:
	def __init__(self):
		self.last = []
		self.last_index = None
		self.count = 0
	
	def check(self, difference):
		k = Configuration.dnfa_window_size
		
		# first lets save the new value
		if len(self.last) < k:
			self.last.append(difference)
			self.last_index = len(self.last) - 1
		else:
			self.last_index = (self.last_index + 1) % k
			self.last[self.last_index] = difference
		
		# checking counts
		self.count += 1
		
		if self.count < Configuration.dnfa_min_frames:
			return False
		
		if self.count >= Configuration.dnfa_max_frames:
			self.count = 0
			return True
		
		# now let's calculate the dynamic threshold
		mi = min(self.last)
		me = np.median(self.last)
		threshold = Configuration.dnfa_min_med_diff_factor * max(me - mi, Configuration.dnfa_min_diff_value) + me
		
		if difference > threshold:
			self.count = 0
			return True
		
		return False

class SkyCamera:
	@staticmethod
	def log(message):
		if Configuration.logging:
			time = Time.now()
			log = open(Configuration.log_file, "a")
			log.write("%s: %s\n" % (time.iso, message))
			log.close()
	
	@staticmethod
	def getBitMask():
		return np.uint8(cv2.imread(Configuration.mask_file, 0) / 255)
	
	@staticmethod
	def getMask():
		return np.float32(SkyCamera.getBitMask())
	
	def __init__(self):
		self.device = 0
		self.channel = 0
		self.capture = cv2.VideoCapture()
		self.control = SkyCameraControl()
		self.night_checker = NightChecker()
		self.night = None
		self.last_image = None
		self.differenceChecker = DynamicDifferenceThreshold()
	
	def checkDaytime(self):
		self.night_checker.now()
		night = self.night_checker.isNight()
		
		if self.night != night:
			self.night = night
		
			if Configuration.control_settings:
				self.control.switchConfiguration(self.night, Configuration.verbose_commands)
	
	def open(self):
		self.capture.open(self.device)
	
	def close(self):
		self.capture.release()
	
	def readNight(self):
		sum_image = None
		count = 0
		
		while True:
			if self.capture.grab():
				if cv2.__version__[0] == '3':
					_, image = self.capture.retrieve(flag = self.channel)
				else:
					_, image = self.capture.retrieve(channel = self.channel)
			
				image1 = np.int32(image)
				
				if self.last_image is not None:
					difference = image1 - self.last_image
				else:
					difference = np.array([0])
				
				difference = float(np.sum(np.abs(difference))) / float(difference.size)
				
				if sum_image is None:
					sum_image = self.last_image
				else:
					sum_image += self.last_image
				
				count += 1
				
				self.last_image = image1
				
				if self.differenceChecker.check(difference):
					SkyCamera.log('Difference: %f %d 1' % (difference, count))
				
					time = Time.now()
					
					self.checkDaytime()
					
					return np.uint8(sum_image / count), time
				else:
					SkyCamera.log('Difference: %f %d 0' % (difference, count))
			else:
				return None, None
	
	def read(self):
		sum_image = None
		image = None
		
		count = Configuration.day_averaging_frames
		
		if self.night:
			count = Configuration.night_averaging_frames
		
		for i in range(count):
			if self.capture.grab():
				if cv2.__version__[0] == '3':
					_, image = self.capture.retrieve(flag = self.channel)
				else:
					_, image = self.capture.retrieve(channel = self.channel)
				
				if sum_image is None:
					sum_image = np.int32(image)
				else:
					sum_image += np.int32(image)
		
		time = Time.now()
		
		self.checkDaytime()
		
		return np.uint8(sum_image / count), time
	
	def oneShot(self):
		self.open()
		
		image, time = self.read()
		
		self.close()
		
		return image, time
	
	@staticmethod
	def saveToFile(image, time, path, sub_directory = True):
		SkyCameraFile.stampImage(image, time)
		
		if sub_directory:
			path = os.path.join(path, str(int(time.mjd)))
			
			if not os.path.exists(path):
				os.mkdir(path)
		
		filename = SkyCameraFile.getFileName(time)
		
		path = os.path.join(path, filename)
		
		cv2.imwrite(path, image)
	
	def captureToFile(self, path, sub_directory = True):
		if self.capture.isOpened():
			if Configuration.dnfa_enabled:
				if self.night:
					image, time = self.readNight()
				else:
					image, time = self.read()
			else:
				image, time = self.read()
		else:
			image, time = self.oneShot()
		
		if image is None:
			return None, None
		
		SkyCamera.saveToFile(image, time, path, sub_directory)
		
		return image, time

if __name__ == '__main__':
	directory = Configuration.default_storage_path
	count = Configuration.frame_count
	
	if len(sys.argv) > 1:
		directory = sys.argv[1]
	
	if len(sys.argv) > 2:
		count = int(sys.argv[2])
	
	window = 'All Sky Camera'
	
	camera = SkyCamera()
	camera.open()
	
	loop = True
	
	if Configuration.show_recorded_frames:
		cv2.namedWindow(window)
	
	i = 0
	
	start = Time.now()
	
	while loop:
		image, time = camera.captureToFile(directory, Configuration.store_in_subdirectory)
		
		if image is None:
			print("Error opening Device")
			sys.exit(1)
		
		print("stored image %d at time %s" % (i, time.iso))
		
		i += 1
		
		if Configuration.show_recorded_frames:
			cv2.imshow(window, image)
		
		time_between_frames = Configuration.day_time_between_frames
		
		if camera.night:
			time_between_frames = Configuration.night_time_between_frames
		
		end = Time.now()
		
		deltatime = int((time_between_frames - (end - start).sec) * 1000)
		
		if deltatime < 1:
			deltatime = 1
		
		key = cv2.waitKey(deltatime) & 0xFF
		
		start = Time.now()
		
		if key == 27:
			loop = False
		
		if count > 0:
			count -= 1
			if count == 0:
				loop = False
	
	camera.close()
	
	if Configuration.show_recorded_frames:
		cv2.destroyAllWindows()
