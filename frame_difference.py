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

import cv2
import sys
import numpy as np
import os.path
import glob
from skycamera import SkyCamera

class FrameDifference:
	def __init__(self):
		self.mask = cv2.cvtColor(SkyCamera.getBitMask(), cv2.COLOR_GRAY2BGR)
	
	def difference(self, image1, image2):
		image1 = np.int32(image1 * self.mask)
		image2 = np.int32(image2 * self.mask)

		self.diff = image1 - image2

	def getValue(self):
		return np.mean(np.abs(self.diff))

	def getImage(self):
		return cv2.cvtColor(np.uint8(np.abs(self.diff)), cv2.COLOR_BGR2GRAY)

class FrameComparison:
	def __init__(self, files):
		if len(files) < 2:
			raise Exception('Need at least two files to compare.')
		
		self.image_window = 'Image'
		self.threshold_window = 'Threshold'
		self.difference_window = 'Difference'
		self.files = files
		self.tb_threshold = 'Threshold'
		self.tb_image = 'Image'
		self.current_image = 0
		
		self.image1 = None
		self.image2 = None
		self.difference = None
		self.threshold = 25
		self.gray = None
		
		cv2.namedWindow(self.image_window, cv2.WINDOW_AUTOSIZE)
		cv2.namedWindow(self.difference_window, cv2.WINDOW_AUTOSIZE)
		cv2.namedWindow(self.threshold_window, cv2.WINDOW_AUTOSIZE)
		cv2.createTrackbar(self.tb_image, self.difference_window, 0, len(self.files) - 2, self.selectImage)
		cv2.createTrackbar(self.tb_threshold, self.threshold_window, self.threshold, 255, self.renderThreshold)
		self.render()
	
	def selectImage(self, number):
		if number >= len(self.files) - 1 or number < 0:
			return
		
		self.current_image = number
		self.render()
	
	def render(self):
		self.image1 = np.int32(cv2.imread(self.files[self.current_image], 1))
		self.image2 = np.int32(cv2.imread(self.files[self.current_image + 1], 1))

		self.difference = self.image1 - self.image2

		self.gray = cv2.cvtColor(np.uint8(np.abs(self.difference)), cv2.COLOR_BGR2GRAY)

		self.difference = np.uint8(self.difference * 2 + 128)
		
		cv2.imshow(self.image_window, np.uint8(self.image1))
		#cv2.imshow(self.difference_window, self.difference)
		cv2.imshow(self.difference_window, self.gray)
		self.renderThreshold(self.threshold)
	
	def renderThreshold(self, threshold):
		self.threshold = threshold
		_, thresh = cv2.threshold(self.gray, threshold, 255, cv2.THRESH_BINARY)
		cv2.imshow(self.threshold_window, thresh)

	def run(self):
		while(True):
			k = cv2.waitKey(30) & 0xFF
			if k == 27:
				break
			if k == ord('s'):
				filename = 'out.png'
				print('Saving ' + filename)
				cv2.imwrite(filename, self.difference)
				
		cv2.destroyAllWindows()


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('Usage: frame_difference <path>...')
		exit(1)

	files = []
	
	for path in sys.argv[1:]:
		if os.path.isfile(path):
			files.append(path)
		elif os.path.isdir(path):
			files += sorted(glob.glob(os.path.join(path, "*.jpg")))
		else:
			print('Invalid parameter: %s' % path)
	
	frame_comparison = FrameComparison(files)
	
	frame_comparison.run()
	
	




#__import__("code").interact(local=locals())

