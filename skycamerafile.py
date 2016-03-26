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
import glob
import os
import cv2

class SkyCameraFile:
	@staticmethod
	def glob(path):
		files = glob.glob(os.path.join(path, "frame_[0-9][0-9][0-9][0-9][0-9]_[0-9][0-9]_[0-9][0-9]_[0-9][0-9].jpg"))
		files.sort()
		return files
		
	@staticmethod
	def uniqueName(path):
		return os.path.splitext(os.path.basename(path))[0]
	
	@staticmethod
	def parseTime(path):
		filename = SkyCameraFile.uniqueName(path)
		parts = filename.split('_')
		if parts[0] == 'frame' and len(parts) == 5:
			jd = int(parts[1])
			hh = int(parts[2])
			mm = int(parts[3])
			ss = int(parts[4])
			mjd = jd + (((ss / 60) + mm) / 60 + hh) / 24
			return Time(mjd, format='mjd', scale='utc')
		else:
			raise Exception('Invalid filename.')
	
	@staticmethod
	def getFileName(when):
		mjd = int(when.mjd)
		time = when.iso.split('.')[0].split(' ')[1].replace(':', '_')
		return "frame_%d_%s.jpg" % (mjd, time)
	
	@staticmethod
	def _stampText(image, text, line):
		font = cv2.FONT_HERSHEY_SIMPLEX
		font_scale = 0.55
		margin = 5
		thickness = 2
		color = (255, 255, 255)

		size = cv2.getTextSize(text, font, font_scale, thickness)

		text_width = size[0][0]
		text_height = size[0][1]
		line_height = text_height + size[1] + margin
		
		x = image.shape[1] - margin - text_width
		y = margin + size[0][1] + line * line_height
		
		cv2.putText(image, text, (x, y), font, font_scale, color, thickness)

	@staticmethod
	def stampImage(image, when):
		mjd = int(when.mjd)
		temp = when.iso.split('.')[0].split(' ')
		date = temp[0]
		time = temp[1]
		
		SkyCameraFile._stampText(image, date, 0)
		SkyCameraFile._stampText(image, "UT " + time, 1)
		SkyCameraFile._stampText(image, "MJD " + str(mjd), 2)
