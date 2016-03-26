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
import numpy as np
from scipy import optimize
from skycamera import SkyCamera
from configuration import Configuration
import time

class GaussianStarFinder:
	def __init__(self):
		self.background_gaussian = None
		self.mask = SkyCamera.getMask()
	
	@staticmethod
	def gaussBivarFit(xy, *p):
		(x, y) = xy
		
		A, x0, y0, v1, v2, v3 = p
		
		#X, Y = np.meshgrid(x - x0, y - y0)
		X, Y = x - x0, y - y0
		
		Z = A * np.exp(-1 / 2 * (v1 * X ** 2 + v2 * X * Y + v3 * Y ** 2))
		
		return Z.ravel()
	
	def setImage(self, image):
		self.image = self.removeBackground(image)
	
	def removeBackground(self, image):
		gray = np.float32(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) / 255
		
		if self.background_gaussian is None or self.background_gaussian.shape[0] != Configuration.gaussian_kernel_size:
			self.background_gaussian = cv2.getGaussianKernel(Configuration.gaussian_kernel_size, -1, cv2.CV_32F)
		
		background = cv2.sepFilter2D(gray, cv2.CV_32F, self.background_gaussian, self.background_gaussian)
		
		result = gray - background
		
		result = result * self.mask
		
		mi = np.min(result)
		ma = np.max(result)
		
		#result = (result - mi) / (ma - mi)
		return result / ma

	def isStar(self, x, y):
		A, _, _, _ = self.findStar(x, y)
		return A >= Configuration.gaussian_threshold
	
	def findStar(self, x, y):
		x = int(x)
		y = int(y)
		
		roi_size = Configuration.gaussian_roi_size
		
		roi = self.image[y - roi_size:y + roi_size + 1, x - roi_size:x + roi_size + 1]
		
		X, Y = np.meshgrid(range(x - roi_size, x + roi_size + 1), range(y - roi_size, y + roi_size + 1))
		
		p0 = (1, x, y, 1, 0, 1)
		
		try:
			popt, _ = optimize.curve_fit(self.gaussBivarFit, (X, Y), roi.ravel(), p0=p0, maxfev=10000)
		except Exception as e:
			return 0, (0, 0), np.matrix([[0, 0], [0, 0]]), roi
		
		A, x0, y0, v1, v2, v3 = popt
		
		cov = np.matrix([[v1, v2 / 2], [v2 / 2, v3]]).I
		mu = (x0, y0)
		
		return A, mu, cov, roi

class CandidateStarFinder:
	def __init__(self, detector):
		self.detector = detector
		
	def setDetector(self, detector):
		self.detector = detector
		
	def setImage(self, image):
		self.image = image
		self.candidates = self.detector.detect(image)
		
	def isStar(self, x, y):
		for pos in self.candidates:
			dx = x - pos[0]
			dy = y - pos[1]
			if dx * dx + dy * dy < Configuration.candidate_radius * Configuration.candidate_radius:
				return True
		
		return False
	
	def drawCandidates(self, image):
		for pos in self.candidates:
			cv2.circle(image, tuple(np.int32(pos)), 3, (0, 0, 255))

class FASTStarDetector:
	def __init__(self):
		self.fast = cv2.FastFeatureDetector_create()
		self.mask = SkyCamera.getMask()
	
	def detect(self, image):
		keypoints = self.fast.detect(image, np.uint8(self.mask))
		
		return [kp.pt for kp in keypoints]
	
class GFTTStarDetector:
	def __init__(self):
		pass
	
	def detect(self, image):
		gray = np.float32(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
		
		corners = cv2.goodFeaturesToTrack(gray, Configuration.gftt_max_corners, Configuration.gftt_quality_level, Configuration.gftt_min_distance)
		
		return [x[0] for x in corners]

class SURFStarDetector:
	def __init__(self):
		self.threshold = Configuration.surf_threshold
		
		self.surf = None
		
		self.mask = SkyCamera.getMask()
	
	def detect(self, image):
		if self.threshold != Configuration.surf_threshold:
			self.surf = None
		
		if self.surf is None:
			self.surf = cv2.xfeatures2d.SURF_create(self.threshold)
			self.surf.setUpright(True)
		
		keypoints = self.surf.detect(image, np.uint8(self.mask))
		
		return [kp.pt for kp in keypoints]

class LoGStarDetector:
	def __init__(self):
		self.gaussian = None

	def detect(self, image):
		floatimage = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2GRAY) / 255
		
		if self.gaussian is None or self.gaussian.shape[0] != Configuration.log_kernel_size:
			self.gaussian = cv2.getGaussianKernel(Configuration.log_kernel_size, -1, cv2.CV_32F)
		
		gaussian_filtered = cv2.sepFilter2D(floatimage, cv2.CV_32F, self.gaussian, self.gaussian)
		
		# LoG
		filtered = cv2.Laplacian(gaussian_filtered, cv2.CV_32F, ksize=Configuration.log_block_size)

		# DoG
		#gaussian2 = cv2.getGaussianKernel(Configuration.log_block_size, -1, cv2.CV_32F)
		#gaussian_filtered2 = cv2.sepFilter2D(floatimage, cv2.CV_32F, gaussian2, gaussian2)
		#filtered = gaussian_filtered - gaussian_filtered2

		mi = np.min(filtered)
		ma = np.max(filtered)
		
		if mi - ma != 0:
			filtered = 1 - (filtered - mi) / (ma - mi)

		_, thresholded = cv2.threshold(filtered, Configuration.log_threshold, 1.0, cv2.THRESH_BINARY)
		self.debug = thresholded
		thresholded = np.uint8(thresholded)
		
		contours = None
		
		if int(cv2.__version__.split('.')[0]) == 2:
			contours, _ = cv2.findContours(thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		else:
			_, contours, _ = cv2.findContours(thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

		candidates = []
		
		for i in range(len(contours)):
			rect = cv2.boundingRect(contours[i])
			v1 = rect[0:2]
			v2 = np.add(rect[0:2], rect[2:4])
			if rect[2] < Configuration.log_max_rect_size and rect[3] < Configuration.log_max_rect_size:
				roi = floatimage[v1[1]:v2[1], v1[0]:v2[0]]
				_, _, _, maxLoc = cv2.minMaxLoc(roi)
				maxLoc = np.add(maxLoc, v1)
				
				candidates.append(maxLoc)
		
		self.candidates = candidates
		
		return candidates

if __name__ == '__main__':
	def nothing(x):
		pass
	
	def hist_lines(image, start, end):
		scale = 4
		height = 1080
		
		result = np.zeros((height, 256 * scale, 1))
		
		hist = cv2.calcHist([image], [0], None, [256], [start, end])
		cv2.normalize(hist, hist, 0, height, cv2.NORM_MINMAX)
		hist = np.int32(np.around(hist))
		
		for x, y in enumerate(hist):
			cv2.rectangle(result, (x * scale, 0), ((x + 1) * scale, y), (255), -1)
			
		result = np.flipud(result)
		return result

	if len(sys.argv) < 2:
		print('Usage: nephoscope <image>')
		sys.exit(1)

	filename = sys.argv[1]

	print('Reading ' + filename)

	image = cv2.imread(filename, 1)

	#image = cv2.fastNlMeansDenoisingColored(image, None, 2, 2, 7, 21)

	window = 'Nephoscope - star detection'
	window2 = 'Histogram'
	tb_image_switch = '0: original\n1: stars\n2: debug\n3: mser'
	tb_kernel_size = 'Kernel size (*2 + 1)'
	tb_block_size = 'Block size (*2 + 1)'
	tb_threshold = 'threshold'

	#import matplotlib
	#matplotlib.use('agg')
	from matplotlib import pyplot as plt

	cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

	cv2.createTrackbar(tb_image_switch, window, 7, 7, nothing)
	cv2.createTrackbar(tb_kernel_size, window, 2, 100, nothing)
	cv2.createTrackbar(tb_block_size, window, 4, 100, nothing)
	cv2.createTrackbar(tb_threshold, window, 53, 100, nothing)

	cv2.imshow(window, image)
	
	cv2.namedWindow(window2, cv2.WINDOW_AUTOSIZE)

	height = image.shape[0]
	width = image.shape[1]

	# compression makes the mask bad, so we throd away the last two bits
	b, g, r = cv2.split(image)
	saturated = np.float32(cv2.bitwise_and(cv2.bitwise_and(b, g), r) > 251)

	gaussian_star_finder = GaussianStarFinder()
	log_star_detector = LoGStarDetector()
	candidate_finder = CandidateStarFinder(log_star_detector)
	candidate_finder.setImage(image)
	
	star_detectors = {}
	
	star_detectors[1] = log_star_detector
	star_detectors[2] = log_star_detector
	star_detectors[4] = GFTTStarDetector()
	star_detectors[5] = SURFStarDetector()
	star_detectors[6] = FASTStarDetector()

	last_image_switch = 0
	
	result = image

	while True:
		image_switch = cv2.getTrackbarPos(tb_image_switch, window)
		kernel_size = cv2.getTrackbarPos(tb_kernel_size, window) * 2 + 1
		block_size = cv2.getTrackbarPos(tb_block_size, window) * 2 + 1
		threshold = cv2.getTrackbarPos(tb_threshold, window) / 100.0

		if image_switch != last_image_switch:
			if image_switch in star_detectors:
				candidate_finder.setDetector(star_detectors[image_switch])
				candidate_finder.setImage(image)
			last_image_switch = image_switch

		if image_switch == 0:
			result = image
		elif image_switch == 3:
			mser = None
			if int(cv2.__version__.split('.')[0]) == 2:
				mser = cv2.MSER(1, 1, 30)
			else:
				mser = cv2.MSER_create(1, 1, 30)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			msers = mser.detect(gray)
			result = image.copy()
			cv2.polylines(result, msers, True, (0, 0, 255))
		elif image_switch == 4:
			result = image.copy()
			
			candidate_finder.drawCandidates(result)
		elif image_switch == 5:
			Configuration.surf_threshold = threshold * 100
			
			candidate_finder.setImage(image)
			
			result = image.copy()
			
			candidate_finder.drawCandidates(result)
		elif image_switch == 6:
			result = image.copy()
			
			candidate_finder.drawCandidates(result)
		elif image_switch == 7:
			result = gaussian_star_finder.removeBackground(image)
			
			gaussian_size = 4
			sigma = 2
			
			gaussian = cv2.getGaussianKernel(gaussian_size * 2 + 1, sigma, cv2.CV_32F)
			
			final = np.outer(gaussian, gaussian)
			
			hist = hist_lines(result, 0.01, 1)
			
			cv2.imshow(window2, hist)
			
			if True:
				#result2 = cv2.matchTemplate(result, final, cv2.TM_SQDIFF_NORMED)
				#result2 = cv2.matchTemplate(result, final, cv2.TM_SQDIFF)
				
				#size = result2.shape
				
				#result = np.zeros(result.shape, np.float32)
				#result[gaussian_size:(gaussian_size + size[0]), gaussian_size:(gaussian_size + size[1])] = 1 - result2
			
				#result = result * gaussian_star_finder.mask
			
				#print(np.min(result))
				#print(np.max(result))
				
				#_, result = cv2.threshold(result, threshold, 1.0, cv2.THRESH_BINARY)
				pass
			else:
				fast = cv2.FastFeatureDetector_create()
				
				kp = fast.detect(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), np.uint8(gaussian_star_finder.mask))
				
				result = image.copy()
				
				cv2.drawKeypoints(result, kp, result, (0,0,255), 1)
		else:
			log_star_detector.kernel_size = kernel_size
			log_star_detector.block_size = block_size
			log_star_detector.threshold = threshold
			
			candidate_finder.setImage(image)
			
			if image_switch == 2:
				result = log_star_detector.debug
				
				masked = cv2.multiply(result, 1 - saturated)
				result = cv2.multiply(masked, gaussian_star_finder.mask) * 255
			else:
				result = image.copy()
				candidate_finder.drawCandidates(result)
			
		cv2.imshow(window, result)


		k = cv2.waitKey(30) & 0xFF
		if k == 27:
			break
		if k == ord('s'):
			filename = 'out.png'
			print('Saving ' + filename)
			cv2.imwrite(filename, result)
		
		if k == ord(' '):
			print(np.max(result))
			print(np.min(result))

	#__import__("code").interact(local=locals())

	cv2.destroyAllWindows()
