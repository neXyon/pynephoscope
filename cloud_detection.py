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
import numpy as np
import skycamera
import pandas as pd
import pickle
from configuration import Configuration
from skycamerafile import *
from calibration import Calibration
from sky import *

class CDRBDifference:
	def detect(self, image, mask = None):
		b,g,r = cv2.split(image)
		
		difference = cv2.subtract(np.float32(r), np.float32(b))
		
		_, result = cv2.threshold(difference, Configuration.rb_difference_threshold, 1, cv2.THRESH_BINARY)
		
		return np.uint8(result)

class CDRBRatio:
	def detect(self, image, mask = None):
		floatimage = np.float32(image)

		fb,fg,fr = cv2.split(floatimage)
		
		nonzero = fb != 0
		difference = np.zeros(fr.shape, np.float32)
		difference[nonzero] = fr[nonzero] / fb[nonzero]
		_, result = cv2.threshold(difference, Configuration.rb_ratio_threshold, 1, cv2.THRESH_BINARY)
		return np.uint8(result)

class CDBRRatio:
	def detect(self, image, mask = None):
		floatimage = np.float32(image)

		fb,fg,fr = cv2.split(floatimage)
		
		nonzero = fr != 0
		difference = np.zeros(fr.shape, np.float32)
		difference[nonzero] = fb[nonzero] / fr[nonzero]
		_, result = cv2.threshold(difference, Configuration.br_ratio_threshold, 1, cv2.THRESH_BINARY_INV)
		return np.uint8(result)

class CDNBRRatio:
	def detect(self, image, mask = None):
		floatimage = np.float32(image)

		fb,fg,fr = cv2.split(floatimage)
		
		nonzero = (fr + fb) != 0
		difference = np.zeros(fr.shape, np.float32)
		difference[nonzero] = (fb[nonzero] - fr[nonzero]) / (fb[nonzero] + fr[nonzero])
		_, result = cv2.threshold(difference, Configuration.nbr_threshold, 1, cv2.THRESH_BINARY_INV)
		return np.uint8(result)

class CDAdaptive:
	def detect(self, image, mask = None):
		b,g,r = cv2.split(image)
		
		difference = cv2.subtract(r, b)
		difference = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		
		# ADAPTIVE_THRESH_GAUSSIAN_C or ADAPTIVE_THRESH_MEAN_C
		return cv2.adaptiveThreshold(difference, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, Configuration.adaptive_block_size, Configuration.adaptive_threshold)

class CDBackground:
	def __init__(self):
		self.kernel = None
		
	def detect(self, image, mask = None):
		floatimage = np.float32(image)

		fb,fg,fr = cv2.split(floatimage)
		
		# red-to-blue channel operation
		ra = fr + fb
		rb = fr - fb
		rb[ra > 0] /= ra[ra > 0]
		#mi = np.min(rb)
		#ma = np.max(rb)
		#rb = np.uint8((rb - mi) / (ma - mi) * 255)
		
		# morphology open
		if self.kernel is None or self.kernel.shape[0] != Configuration.background_rect_size:
			self.kernel = np.ones((Configuration.background_rect_size, Configuration.background_rect_size), np.uint8) * 255
		
		result = cv2.morphologyEx(rb, cv2.MORPH_OPEN, self.kernel)
		
		# background subtraction
		# homogeneous background image V
		result = rb - result
		
		mi = np.min(result)
		ma = np.max(result)
		result = np.uint8((result - mi) / (ma - mi) * 255)
		
		# adaptive threshold T
		T, _ = cv2.threshold(result[mask == 0], 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		
		# V(i, j) > T
		return np.uint8((T - np.float32(result)) <= 0)

class CDMulticolor:
	def detect(self, image, mask = None):
		b,g,r = cv2.split(image)
		
		return np.uint8((b < r + Configuration.mc_rb_threshold) & (b < g + Configuration.mc_bg_threshold) & (b < Configuration.mc_b_threshold))

class CDSuperPixel:
	def __init__(self):
		self.width = None
		self.height = None
		self.seeds = None
		self.threshold = None
		
		self.reset()
		
		self.gaussian = None
	
	def reset(self):
		self.num_superpixels = Configuration.sp_num_superpixels
		self.prior = Configuration.sp_prior
		self.num_levels = Configuration.sp_num_levels
		self.num_histogram_bins = Configuration.sp_num_histogram_bins
	
	def detect(self, image, mask = None):
		if self.width != image.shape[1] or self.height != image.shape[0] or self.channels != image.shape[2]:
			self.seeds = None
		
		if self.num_superpixels != Configuration.sp_num_superpixels or self.prior != Configuration.sp_prior or self.num_levels != Configuration.sp_num_levels or self.num_histogram_bins != Configuration.sp_num_histogram_bins:
			self.seeds = None
			self.reset()
		
		if self.seeds is None:
			self.width = image.shape[1]
			self.height = image.shape[0]
			self.channels = image.shape[2]
			self.seeds = cv2.ximgproc.createSuperpixelSEEDS(self.width, self.height, self.channels, Configuration.sp_num_superpixels, Configuration.sp_num_levels, Configuration.sp_prior, Configuration.sp_num_histogram_bins)
			self.threshold = np.ones((self.height, self.width), np.float32)
		
		converted_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		
		self.seeds.iterate(converted_img, Configuration.sp_num_iterations)
		labels = self.seeds.getLabels()
		
		if mask is None:
			mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
		
		floatimage = np.float32(image)
		fb, fg, fr = cv2.split(floatimage)
		
		rb = fr - fb
		#rb = fr + fb + fg
		mi = np.min(rb[mask == 0])
		ma = np.max(rb[mask == 0])
		rb = np.uint8((rb - mi) / (ma - mi) * 255)
		
		#mimaTg = np.uint8((np.array([-15, 15]) - mi) / (ma - mi) * 255)
		
		Tg, _ = cv2.threshold(rb[mask == 0], 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		
		#if Tg < mimaTg[0]:
		#	Tg = mimaTg[0]
		#elif Tg > mimaTg[1]:
		#	Tg = mimaTg[1]
		
		Tl = np.zeros(self.seeds.getNumberOfSuperpixels())
		
		for i in range(self.seeds.getNumberOfSuperpixels()):
			sp = rb[(labels == i) & (mask == 0)]
			if sp.size == 0:
				self.threshold[labels == i] = Tg
				continue
			
			Lmax = np.max(sp)
			Lmin = np.min(sp)
			if Lmax < Tg:
				Tl[i] = Tg#Lmax
			elif Lmin > Tg:
				Tl[i] = Tg#Lmin
			else:
				Sl, _ = cv2.threshold(sp, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
				Tl[i] = 0.5 * (Sl + Tg)
			
			self.threshold[labels == i] = Tl[i]
		
		if self.gaussian is None or self.gaussian.shape[0] != Configuration.sp_kernel_size:
			self.gaussian = cv2.getGaussianKernel(Configuration.sp_kernel_size, -1, cv2.CV_32F)
		
		self.threshold = cv2.sepFilter2D(self.threshold, cv2.CV_32F, self.gaussian, self.gaussian)
		
		return np.uint8((self.threshold - rb) <= 0)# * mask

class CDSunRemoval:
	@staticmethod
	def circle_mask(pos, radius, shape):
		cy, cx = pos
		x, y = np.ogrid[:shape[0], :shape[1]]
		return (x - cx)*(x - cx) + (y - cy)*(y - cy) < radius*radius
	
	@staticmethod
	def find_sun(start_pos, mask, static_mask=None):
		x = int(start_pos[0])
		y = int(start_pos[1])
		
		pos = start_pos
		radius = 1

		# fully white images
		if np.sum(mask == 0) < 100:
			return None, None, None
		
		# exception thrown when index out of bounds
		if x < 0 or y < 0 or x >= mask.shape[1] or y >= mask.shape[0]:
			#print('Coordinates not in image')
			#return None, None, None
			circle = 0
			while(np.sum(circle) == 0):
				circle = CDSunRemoval.circle_mask(pos, radius, mask.shape)
				radius += 2
		else:
			# Sun not found
			if mask[y, x] == 0:
				#print('Coordinates not saturated')
				return None, None, None
		
		# fix mask at border and get rid of tiny holes
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.array(CDSunRemoval.circle_mask((5, 5), 5.1, (11, 11)), np.uint8))

		# index grid used for centering
		index_grid_x, index_grid_y = np.mgrid[:mask.shape[0], :mask.shape[1]]
		
		# update function when parameters change
		def _update_find_sun(pos, radius, mask):
			circle = CDSunRemoval.circle_mask(pos, radius, mask.shape)
			weights = np.logical_and(mask, circle)
			score = np.sum(weights) / np.sum(circle)

			return circle, weights, score
		
		def _get_outside(pos, radius, shape):
			cy, cx = pos
			x, y = np.mgrid[np.floor(cx - radius):np.ceil(cx + radius), np.floor(cy - radius):np.ceil(cy + radius)]
			res = np.logical_and((x - cx)*(x - cx) + (y - cy)*(y - cy) < radius*radius, np.logical_or(np.logical_or(x < 0, x >= shape[0]), np.logical_or(y < 0, y >= shape[1])))
			return x[res], y[res]
		
		# first phase: radius doubling
		score = 1
		while score == 1:
			radius *= 2
			circle, weights, score = _update_find_sun(pos, radius, mask)
		
		# second phase: radius binary search
		hi = radius
		lo = radius / 2
		
		while hi - lo > 1:
			radius = np.round((hi + lo) / 2)
			circle, weights, score = _update_find_sun(pos, radius, mask)
			if score == 1:
				lo = radius
			else:
				hi = radius
		
		radius = lo

		# if the score is 0 now in the static mask, we give up
		if static_mask is None:
			static_mask = CloudDetectionHelper().mask

		static_score = np.sum(np.logical_and(mask + static_mask - 1, circle))

		if static_score == 0:
			return None, None, None

		old_old_radius = radius - 1
		circle_sum_max = 0
		circle_sum_params = (pos, radius)
		circle_sum_iter = 0
		circle_sum_max_iter = 5
		
		while old_old_radius < radius and circle_sum_iter < circle_sum_max_iter:
			old_old_radius = radius
			old_params = (pos, radius)
		
			# third phase: centering

			old_score = score - 0.1
			while old_score < score:
				old_score = score
				old_pos = pos
				# getting outside for circles that are not fully inside the picture
				ox, oy = _get_outside(pos, radius, mask.shape)
				pos = (np.mean(np.concatenate((oy, index_grid_y[weights]))), np.mean(np.concatenate((index_grid_x[weights], ox))))
				circle, weights, score = _update_find_sun(pos, radius, mask)

			pos = old_pos

			# fourth phase: radius increment

			old_radius = radius
			while score > 0.99:
				old_radius = radius
				radius += 1
				circle, weights, score = _update_find_sun(pos, radius, mask)

			radius = old_radius
			
			circle_sum_iter += 1
			circle_sum = np.sum(circle)
			if circle_sum > circle_sum_max:
				circle_sum_max = circle_sum
				circle_sum_params = (pos, radius)
				circle_sum_iter = 0
		
		if circle_sum_iter >= circle_sum_max_iter:
			pos, radius = circle_sum_params
		else:
			pos, radius = old_params
			
		if not np.isscalar(pos[0]):
			pos = (pos[0][0], pos[1][0])
		
		return CDSunRemoval.circle_mask(pos, radius, mask.shape), pos, radius

	@staticmethod
	def find_sun_line(image, pos):
		# we start off with converting the image to a gray brightness image
		
		res = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
		# parameters:
		
		search_area = 20
		invalid_area = 4
		canny_threshold = 25
		ratio = 3 # 2-3
		hough_threshold = 10
		
		# first determine the center line, if there is one, by finding the maximum median (or if saturated the mean)
		
		max_mean = 0
		max_med = 0
		max_med_pos = pos[0]

		min_med = 255
		
		for x in range(np.max([0, int(pos[0]) - search_area]), np.min([image.shape[1], int(pos[0]) + search_area + 1])):
			mean = np.mean(res[:, x])
			med = np.median(res[:, x])
			if med == max_med or med > 250:
				max_med = med
				if mean > max_mean:
					max_mean = mean
					max_med_pos = x
			elif med > max_med:
				max_med = med
				max_med_pos = x
				max_brightness = mean
			if med < min_med:
				min_med = med
		
		# check if there is a line expecting a minimum value for the maximal median
		#  and the position of it close enough to the sun
		
		if max_med < 200 or np.abs(max_med_pos - pos[0]) > search_area - invalid_area or min_med + 25 > max_med:
			return None, None, None
	
		# now we detect all edge lines via canny edge detection and hough transform for lines
	
		res = cv2.blur(res, (3, 3))
		edges = cv2.Canny(res, canny_threshold, canny_threshold * ratio)
	
		lines = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_threshold, minLineLength=20)
	
		# to determine the width of our broad line we now check the detected edge lines close to the center line
	
		min_x = max_med_pos
		max_x = max_med_pos
	
		for line in lines:
			line = line[0]
	
			# we use only vertical lines
	
			if line[0] - line[2] == 0:
				x = line[0]
	
				# filter out the antenna at 284-298
	
				if x >= 284 and x <= 298:
					if line[1] < 160 and line[3] < 160 and np.abs(x - max_med_pos) > invalid_area:
						continue
	
				# filter out the center line artifact at 318 and 321
	
				if x == 318 or x == 321:
					if np.abs(x - max_med_pos) > invalid_area:
						continue
	
				# take close lines to broaden
	
				if x < min_x and x > max_med_pos - search_area:
					min_x = x
				elif x > max_x and x < max_med_pos + search_area:
					max_x = x
	
		# return the line mask
					
		result = np.zeros(res.shape, np.uint8)
		result[:, min_x:max_x + 1] = True
		return result, min_x, max_x
	
class CloudDetectionHelper:
	def __init__(self):
		self.mask = skycamera.SkyCamera.getBitMask()
		
		self.kernel = None
	
	def get_mask(self, image):
		b,g,r = cv2.split(image)
		
		if self.kernel is None or self.kernel.shape[0] != Configuration.morphology_kernel_size:
			self.kernel = np.ones((Configuration.morphology_kernel_size, Configuration.morphology_kernel_size), np.uint8)
		
		saturated = np.uint8(cv2.bitwise_and(cv2.bitwise_and(b, g), r) > 254)
		saturated = cv2.dilate(saturated, self.kernel, iterations = Configuration.morphology_iterations)

		self.fullmask = np.uint8(np.logical_or(np.logical_not(self.mask), saturated))

		return self.fullmask
	
	def close_result(self, result):
		return cv2.morphologyEx(result, cv2.MORPH_CLOSE, self.kernel)
	
	def get_result_image(self, result):
		result_image = result * 255
		
		result_image[self.fullmask != 0] = 128
		
		return result_image
	
	def get_cloudiness(self, result):
		usable_part = result[self.fullmask == 0]
		
		return np.sum(usable_part) / usable_part.size
		
	def get_unsaturated(self):
		return np.sum(self.fullmask == 0) / np.sum(self.mask != 0)


class CDSST:
    def __init__(self):
        path = 'frames/'
        day_images = SkyCameraFile.glob(path)
        day_images.sort()
        times = np.array([SkyCameraFile.parseTime(x).datetime for x in day_images])
        
        self.day_images = day_images
        self.times = times
        
        self.calibration = Calibration(catalog=SkyCatalog(True))
        self.calibration.load()

        self.helper = CloudDetectionHelper()
        
        try:
            with open('cd_sst.cache', 'rb') as f:
                self.cache = pickle.load(f)
        except:
            self.cache = pd.DataFrame(index=pd.Index([], dtype=np.datetime64), columns=['pos_x', 'pos_y', 'radius', 'stripe_min', 'stripe_max'])
        
    def save_cache(self):
        with open('cd_sst.cache', 'wb') as f:
            pickle.dump(self.cache, f)
        
    def detect(self, filename):
        image = cv2.imread(filename)
        time = SkyCameraFile.parseTime(filename).datetime
        
        mask = np.uint8(self.helper.get_mask(image).copy())

        self.calibration.selectImage(filename)
        pos = self.calibration.project()
        pos = (pos[0], pos[1])

        if time in self.cache.index:
            sun_x, sun_y, radius, min_x, max_x = self.cache.loc[time]
            
            sun = None
            sun_pos = None
            
            if not np.isnan(sun_x):
                sun_pos = (sun_x, sun_y)
                sun = CDSunRemoval.circle_mask(sun_pos, radius, mask.shape)
                
            sun_line = None
            if not np.isnan(min_x):
                sun_line = np.zeros(mask.shape, np.uint8)
                sun_line[:, min_x:max_x + 1] = True
            
        else:
            sun_line, min_x, max_x = CDSunRemoval.find_sun_line(image, pos)

            sun, sun_pos, radius = CDSunRemoval.find_sun(pos, mask)
            
            sun_x = sun_y = None
            if sun_pos is not None:
                sun_x = sun_pos[0]
                sun_y = sun_pos[1]
            
            self.cache.loc[time] = [sun_x, sun_y, radius, min_x, max_x]

        if sun_pos is None:
            sun_pos = pos

        mask = self.helper.fullmask.copy()
        mask[self.helper.mask == 0] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

        _, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        cloudiness = np.ones(mask.shape, np.float32) * .5

        did_sun = False

        for contour in contours:
            area = cv2.contourArea(contour)
            is_sun = False

            if area > 100: # TODO: try different numbers
                single_contour = np.zeros(mask.shape, np.uint8)
                cv2.drawContours(single_contour, [contour], 0, 1, cv2.FILLED)

                if sun is not None and not did_sun:
                    sun_area = np.sum(sun[self.helper.mask == 1])

                    if area > 0.9 * sun_area:
                        joint_area = np.sum(np.logical_and(single_contour, sun))

                        if sun_area / joint_area > 0.9:
                            is_sun = True

                    if is_sun:
                        if sun_area * 1.2 < area:
                            difference = np.uint8(np.logical_and(np.logical_not(sun), single_contour))
                            _, contours2, _ = cv2.findContours(difference, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                            # filter smaller ones here! currently done with the if at the beginning
                            contours += contours2
                            did_sun = True

                if not is_sun:
                    cloudiness[single_contour > 0] = 1.0



        b, g, r = cv2.split(np.int32(image))

        mask = self.helper.mask

        rmb = r - b
        rmb[mask == 0] = 0

        cloudiness[rmb < -10] = 0
        cloudiness[rmb > 50] = 1

        delta = np.timedelta64(39, 's')
        delta2 = np.timedelta64(0, 's')
        time_diff = time - self.times
        before = np.logical_and(time_diff > delta2, time_diff < delta)

        if np.sum(before) == 0:
            raise ValueError('No previous image found.')

        current_index = np.where(before)[0][0]

        prev_img = cv2.imread(self.day_images[current_index])

        gray_prev = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_image, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow2 = -cv2.calcOpticalFlowFarneback(gray_image, gray_prev, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag1, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
        mag2, _ = cv2.cartToPolar(flow2[...,0], flow2[...,1])

        movement = np.logical_and(mag1 > 1, mag2 > 1)
        no_movement = np.logical_not(movement)

        brightness = np.mean(image, 2)

        cloudiness[np.logical_and(movement == 1, cloudiness != 0)] = 1
        cloudiness[self.helper.mask == 0] = 1
        if sun is not None:
            cloudiness[sun == 1] = 1

        if sun_line is not None:
            sun_line_dilated = cv2.morphologyEx(sun_line, cv2.MORPH_DILATE, np.ones((1, 3)))
            cloudiness[sun_line_dilated == 1] = 1

        y, x = np.mgrid[0:brightness.shape[0], 0:brightness.shape[1]]

        x1 = []
        y1 = []
        out = []

        for i in range(brightness.shape[0]):
            for j in range(brightness.shape[1]):
                if cloudiness[i, j] != 1:
                    x1.append(x[i, j])
                    y1.append(y[i, j])
                    out.append(brightness[i, j])

        x = np.array(x1) - sun_pos[0]
        y = np.array(y1) - sun_pos[1]

        out = np.array(out)

        dist = np.sqrt(x * x + y * y)

        A = np.array([dist, np.ones(x.shape), x, y]).transpose()
        A_inv = np.linalg.pinv(A)

        param = np.dot(A_inv, out)

        y, x = np.mgrid[0:brightness.shape[0], 0:brightness.shape[1]]
        x = x - pos[0]
        y = y - pos[1]
        dist = np.sqrt(x * x + y * y)
        A = np.array([dist, np.ones(x.shape), x, y]).transpose()
        gradient = np.dot(A, param).transpose()

        rect_size = 15

        rect_border = (rect_size - 1) // 2

        brightness_norm = brightness - gradient

        stddev = np.zeros(brightness.shape)

        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if cloudiness[y, x] == 1:
                    continue

                lx = x - rect_border
                rx = x + rect_border + 1
                uy = y - rect_border
                dy = y + rect_border + 1

                if lx < 0: lx = 0
                if uy < 0: uy = 0
                if rx > image.shape[1]: rx = image.shape[1]
                if dy > image.shape[0]: dy = image.shape[0]
                    
                mask_part = cloudiness[uy:dy, lx:rx]
                stddev[y, x] = np.std(brightness_norm[uy:dy, lx:rx][mask_part != 1])

        def_clear = np.sum(cloudiness == 0)

        cloudiness[cloudiness == 0.5] = (stddev > 3)[cloudiness == 0.5]

        if sun is None or (sun_line is None and radius < 100):
            if def_clear < 0.1 * np.sum(self.helper.mask == 1):
                cloudiness[np.logical_and(cloudiness == 0, rmb > -8)] = 1
        
        cloudiness = self.helper.close_result(cloudiness)

        cloudiness[self.helper.mask == 0] = 0.5

        if sun is not None:
            cloudiness[sun == 1] = 0.5

        if sun_line is not None:
            cloudiness[sun_line_dilated == 1] = 0.5

        return cloudiness

    def get_cloud_cover(self, cloudiness):
        return np.sum(cloudiness == 1) / np.sum(cloudiness != 0.5)

if __name__ == '__main__':
	import sys
	
	if len(sys.argv) < 2:
		print('Usage: cloude_detection <image>')
		sys.exit(1)
		
	filename = sys.argv[1]
	
	image = cv2.imread(filename, 1)
	
	#detector = CDRBDifference()
	#detector = CDRBRatio()
	#detector = CDBRRatio()
	#detector = CDAdaptive()
	#detector = CDBackground()
	#detector = CDMulticolor()
	detector = CDSuperPixel()
	helper = CloudDetectionHelper()
	
	result = helper.close_result(detector.detect(image, helper.get_mask(image)))
	
	#cv2.imwrite("result.png", helper.get_result_image(result))
	
	print(helper.get_cloudiness(result), helper.get_unsaturated())
