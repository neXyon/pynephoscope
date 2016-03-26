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
from astropy.coordinates import EarthLocation
from astropy import units as u
from astropy.time import Time
from sky import SkyCatalog
from configuration import Configuration
from skycamerafile import SkyCameraFile
from glob import glob

class MoonChecker:
	def __init__(self):
		location = EarthLocation(lat=Configuration.latitude, lon=Configuration.longitude, height=Configuration.elevation)

		self.catalog = SkyCatalog(False, True)
		self.catalog.setLocation(location)
		self.now()

	def now(self):
		self.setTime(Time.now())

	def setTime(self, time):
		self.catalog.setTime(time)
		_, _, alt, _ = self.catalog.calculate()

		self.alt = alt[0]

		return self.alt
		
	def isUp(self):
		return self.alt > Configuration.moon_up_angle

	def __str__(self):
		if self.isUp():
			return "Up"
		else:
			return "Down"

if __name__ == '__main__':
	checker = MoonChecker()

	if len(sys.argv) > 1:
		files = SkyCameraFile.glob(sys.argv[1])

		for f in files:
			time = SkyCameraFile.parseTime(f)
			checker.setTime(time)
			print(f, checker)
	else:
		print(checker)
