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

from calibration import Calibration
from configuration import Configuration
from star_detection import GaussianStarFinder
from astropy.coordinates import EarthLocation
from astropy import units as u
from astropy.utils.exceptions import AstropyWarning
from scipy.linalg import sqrtm
import numpy as np
import sys
import os
import cv2
import warnings

class StarCheckerHelper:
	def __init__(self, calibration_file):
		self.calibration = Calibration()
		self.calibration.load(calibration_file)
	
	def prepare(self, path, star_finder):
		with warnings.catch_warnings():
			warnings.simplefilter('ignore', AstropyWarning)
			self.calibration.selectImage(path)
		
		self.names, self.vmag, alt, az = self.calibration.catalog.filter(Configuration.min_alt * u.deg, Configuration.max_mag)
		
		altaz = np.array([alt.radian, az.radian]).transpose()
		
		self.pos = np.column_stack(self.calibration.project(altaz))
		
		self.finder = star_finder
		
		self.image = cv2.imread(path)
		
		self.finder.setImage(self.image)
		
		self.altaz = np.array([alt.degree, az.degree]).transpose()
	
	def count_stars(self):
		min_az = 0
		max_az = 360
		min_alt = Configuration.min_alt
		max_alt = 90
		alt_step = Configuration.alt_step
		az_step = Configuration.az_step
		
		alt_bins = int((max_alt - min_alt) / alt_step)
		az_bins = int((max_az - min_az) / az_step)
		
		counts = np.zeros([alt_bins, az_bins, 4])
		
		for alt_bin in range(alt_bins):
			alt = min_alt + alt_step * alt_bin
			for az_bin in range(az_bins):
				az = min_az + az_step * az_bin
				
				counts[alt_bin, az_bin, 2] = alt
				counts[alt_bin, az_bin, 3] = az
				
				for i in range(self.pos.shape[0]):
					aa = self.altaz[i]
					if aa[0] > alt and aa[0] <= alt + alt_step and aa[1] > az and aa[1] <= az + az_step:
						counts[alt_bin, az_bin, 0] += 1
						if self.finder.isStar(self.pos[i][0], self.pos[i][1]):
							counts[alt_bin, az_bin, 1] += 1
		
		return counts
	
	def get_image(self):
		result = self.image.copy()
		
		good_color = (0, 255, 0)
		bad_color = (0, 0, 255)
		
		for i in range(self.pos.shape[0]):
			if self.finder.isStar(self.pos[i][0], self.pos[i][1]):
				color = good_color
			else:
				color = bad_color
			cv2.circle(result, (int(self.pos[i][0]), int(self.pos[i][1])), 3, color)
		
		return result

def renderStarGauss(image, cov, mu, first, scale = 5):
	num_circles = 3
	num_points = 64
	
	cov = sqrtm(cov)
	
	num = num_circles * num_points
	pos = np.ones((num, 2))
	
	for c in range(num_circles):
		r = c + 1
		for p in range(num_points):
			angle = p / num_points * 2 * np.pi
			index = c * num_points + p
			
			x = r * np.cos(angle)
			y = r * np.sin(angle)
			
			pos[index, 0] = x * cov[0, 0] + y * cov[0, 1] + mu[0]
			pos[index, 1] = x * cov[1, 0] + y * cov[1, 1] + mu[1]
	
	#image = image.copy()
	#image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	
	if first:
		image = cv2.resize(image, (0, 0), None, scale, scale, cv2.INTER_NEAREST)
	
	for c in range(num_circles):
		pts = np.array(pos[c * num_points:(c + 1) * num_points, :] * scale + scale / 2, np.int32)
		pts = pts.reshape((-1,1,2))
		cv2.polylines(image, [pts], True, (255, 0, 0))
	
	return image

if __name__ == '__main__':
	if len(sys.argv) < 3:
		print('Usage: star_checker <image> <calibration>')
		sys.exit(1)
	
	roi_size = 5
	
	scale = 5
	circle_radius = 5
	transformed_color = (0, 0, 0)
	good_color = (0, 255, 0)
	bad_color = (0, 0, 255)
	
	window = 'Star Checker'
	
	cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
	
	path = sys.argv[1]
	
	star_finder = GaussianStarFinder()
	
	helper = StarCheckerHelper(sys.argv[2])
	helper.prepare(path, star_finder)
	
	altaz = helper.altaz
	pos = helper.pos
	vmag = helper.vmag
	
	counts = helper.count_stars()
	
	bigimage = helper.image
	gray = cv2.cvtColor(bigimage, cv2.COLOR_BGR2GRAY)
	
	first = True
	
	for alt in range(counts.shape[0]):
		for az in range(counts.shape[1]):
			c = counts[alt, az, 0:4]
			
			if c[0] != 0:
				print(c[2], c[3], c[1] / c[0], c[1], c[0])
			else:
				print(c[2], c[3], c[1], c[1], c[0])
	
	for i in range(pos.shape[0]):
		A, mu, cov, roi = star_finder.findStar(pos[i][0], pos[i][1])
		
		if mu[0] == 0 and mu[1] == 0:
			continue
		
		bigimage = renderStarGauss(bigimage, cov, mu, first, scale)
		first = False
		
		cv2.circle(bigimage, tuple(np.int32((np.int32(pos[i]) + np.array((0.5, 0.5))) * scale)), int(2 * circle_radius - vmag[i]), transformed_color)
		
		(x, y) = np.int32(mu)
		
		roi2 = gray[y - roi_size:y + roi_size + 1, x - roi_size:x + roi_size + 1]
		
		if A < 0.2:
			color = bad_color
		else:
			color = good_color
			print(vmag[i], np.sum(roi), A, (np.sum(roi2) / (2 * roi_size + 1) / (2 * roi_size + 1) - np.min(roi2)) / 255, np.sum(roi2), altaz[i, 0], altaz[i, 1])
		
		cv2.circle(bigimage, tuple(np.int32((np.array(mu) + np.array((0.5, 0.5))) * scale)), int(5 * circle_radius - 4 * vmag[i]), color)
	
	#__import__("code").interact(local=locals())
	
	while True:
		cv2.imshow(window, bigimage)
		
		k = cv2.waitKey(30) & 0xFF
		if k == 27:
			break
		elif k == ord('s'):
			cv2.imwrite('out.png', bigimage)
	
	
	
