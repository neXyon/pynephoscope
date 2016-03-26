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

import numpy as np
import cv2
import ephem
from astropy import units as u
from astropy.time import Time
from astropy.io import ascii
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import Latitude, Longitude
from astropy.table import Column
from skycamerafile import SkyCameraFile
from configuration import Configuration

class SkyCatalog:
	def __init__(self, sun_only = False, moon_only = False):
		if sun_only:
			self.ephemerides = [ephem.Sun()]
			self.data = None
		elif moon_only:
			self.ephemerides = [ephem.Moon()]
			self.data = None
		else:
			self.ephemerides = [ephem.Venus(), ephem.Mars(), ephem.Jupiter(), ephem.Saturn(), ephem.Moon(), ephem.Sun()]
			
			self.data = ascii.read(Configuration.star_catalog_file, guess=False, format='fixed_width_no_header', names=('HR', 'Name', 'DM', 'HD', 'SAO', 'FK5', 'IRflag', 'r_IRflag', 'Multiple', 'ADS', 'ADScomp', 'VarID', 'RAh1900', 'RAm1900', 'RAs1900', 'DE-1900', 'DEd1900', 'DEm1900', 'DEs1900', 'RAh', 'RAm', 'RAs', 'DE-', 'DEd', 'DEm', 'DEs', 'GLON', 'GLAT', 'Vmag', 'n_Vmag', 'u_Vmag', 'B-V', 'u_B-V', 'U-B', 'u_U-B', 'R-I', 'n_R-I', 'SpType', 'n_SpType', 'pmRA', 'pmDE', 'n_Parallax', 'Parallax', 'RadVel', 'n_RadVel', 'l_RotVel', 'RotVel', 'u_RotVel', 'Dmag', 'Sep', 'MultID', 'MultCnt', 'NoteFlag'), col_starts=(0, 4, 14, 25, 31, 37, 41, 42, 43, 44, 49, 51, 60, 62, 64, 68, 69, 71, 73, 75, 77, 79, 83, 84, 86, 88, 90, 96, 102, 107, 108, 109, 114, 115, 120, 121, 126, 127, 147, 148, 154, 160, 161, 166, 170, 174, 176, 179, 180, 184, 190, 194, 196), col_ends=(3, 13, 24, 30, 36, 40, 41, 42, 43, 48, 50, 59, 61, 63, 67, 68, 70, 72, 74, 76, 78, 82, 83, 85, 87, 89, 95, 101, 106, 107, 108, 113, 114, 119, 120, 125, 126, 146, 147, 153, 159, 160, 165, 169, 173, 175, 178, 179, 183, 189, 193, 195, 196))
			
			# removed masked rows
			
			self.data = self.data[:][~np.ma.getmaskarray(self.data['DE-'])]
		
	def setLocation(self, location):
		self.location = location
		
	def setTime(self, time):
		self.time = time
		
	def calculate(self):
		ephem_location = ephem.Observer()
		ephem_location.lat = self.location.latitude.to(u.rad) / u.rad
		ephem_location.lon = self.location.longitude.to(u.rad) / u.rad
		ephem_location.elevation = self.location.height / u.meter
		ephem_location.date = ephem.Date(self.time.datetime)

		if self.data is None:
			self.alt = Latitude([], unit=u.deg)
			self.az = Longitude([], unit=u.deg)
			self.names = Column([], dtype=np.str)
			self.vmag = Column([])
		else:
			ra = Longitude((self.data['RAh'], self.data['RAm'], self.data['RAs']), u.h)
			dec = Latitude((np.core.defchararray.add(self.data['DE-'], self.data['DEd'].astype(str)).astype(int), self.data['DEm'], self.data['DEs']), u.deg)
			c = SkyCoord(ra, dec, frame='icrs')
			altaz = c.transform_to(AltAz(obstime=self.time, location=self.location))
			self.alt = altaz.alt
			self.az = altaz.az

			self.names = self.data['Name']
			self.vmag = self.data['Vmag']

		for ephemeris in self.ephemerides:
			ephemeris.compute(ephem_location)
			self.vmag = self.vmag.insert(0, ephemeris.mag)
			self.alt = self.alt.insert(0, (ephemeris.alt.znorm * u.rad).to(u.deg))
			self.az = self.az.insert(0, (ephemeris.az * u.rad).to(u.deg))
			self.names = self.names.insert(0, ephemeris.name)
		
		return self.names, self.vmag, self.alt, self.az

	def filter(self, min_alt, max_mag):
		show = self.alt >= min_alt

		names = self.names[show]
		vmag = self.vmag[show]
		alt = self.alt[show]
		az = self.az[show]

		show_mags = vmag < max_mag

		names = names[show_mags]
		vmag = vmag[show_mags]
		alt = alt[show_mags]
		az = az[show_mags]
		
		return names, vmag, alt, az

class SkyRenderer:
	def __init__(self, size):
		self.size = size
		self.image = None
		self.font = cv2.FONT_HERSHEY_SIMPLEX
	
	def renderCatalog(self, catalog, max_mag):
		self.names, self.vmag, self.alt, self.az = catalog.filter(0, max_mag)
		return self.render()
	
	def altazToPos(self, altaz):
		if not isinstance(altaz, np.ndarray):
			altaz = np.array([a.radian for a in altaz])
		
		if len(altaz.shape) == 1:
			altaz = np.array([altaz])
		
		r = (1 - altaz[:, 0] / (np.pi / 2)) * self.size / 2
		
		pos = np.ones(altaz.shape)
		
		pos[:, 0] = r * np.cos(-np.pi / 2 - altaz[:, 1]) + self.size / 2
		pos[:, 1] = r * np.sin(-np.pi / 2 - altaz[:, 1]) + self.size / 2

		return pos
	
	def render(self):
		self.image = np.zeros((self.size, self.size, 1), np.uint8)
		
		pos = self.altazToPos(np.column_stack((self.alt.radian, self.az.radian)))
		self.x = pos[:, 0]
		self.y = pos[:, 1]

		for i in range(len(self.names)):
			cv2.circle(self.image, (int(self.x[i]), int(self.y[i])), int(7 - self.vmag[i]), 255, -1)

			if self.vmag[i] < 1 and not np.ma.is_masked(self.names[i]):
				cv2.putText(self.image, self.names[i],(int(self.x[i]) + 6, int(self.y[i])), self.font, 1, 150)
		
		return self.image
	
	def findStar(self, x, y, radius):
		r2 = radius * radius
		
		a = np.subtract(self.x, x)
		b = np.subtract(self.y, y)
		c = a * a + b * b
		
		index = c.argmin()
		if c[index] < radius * radius:
			return (self.alt[index], self.az[index], self.names[index])
		
		return None
	
	def highlightStar(self, image, altaz, radius, color):
		pos = self.altazToPos(altaz)[0]
		
		pos = (int(pos[0]), int(pos[1]))
		
		cv2.circle(image, pos, radius, color)

if __name__ == '__main__':
	import sys
	window = 'Stars'
	size = 1024

	location = EarthLocation(lat=Configuration.latitude, lon=Configuration.longitude, height=Configuration.elevation)
	time = Time.now()

	if len(sys.argv) >= 2:
		try:
			time = SkyCameraFile.parseTime(sys.argv[1])
		except:
			pass

	catalog = SkyCatalog()
	catalog.setLocation(location)
	catalog.setTime(time)
	catalog.calculate()

	renderer = SkyRenderer(size)
	image = renderer.renderCatalog(catalog, 5)

	cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
	cv2.imshow(window, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
