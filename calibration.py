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
import os
import cv2
import pickle
import numpy as np

from scipy.optimize import minimize

from skycamerafile import SkyCameraFile
from sky import SkyCatalog, SkyRenderer
from astropy.coordinates import EarthLocation, AltAz
from astropy import units as u
from configuration import Configuration

class StarCorrespondence:
	def __init__(self, pos, altaz):
		self.pos = pos
		self.altaz = altaz

class Projector:
	def project(self, altaz, k):
		phi = altaz[:, 1] + k[2]
		
		cpsi = np.cos(altaz[:, 0])
		spsi = np.sin(altaz[:, 0])
		cphi = np.cos(phi)
		sphi = np.sin(phi)
		
		ck4 = np.cos(k[3])
		sk4 = np.sin(k[3])
		
		rho_nom = ck4 * cpsi * sphi - sk4 * spsi
		rho_den = cpsi * cphi
		rho = np.arctan2(rho_nom, rho_den)
		
		crho = np.cos(rho)
		
		tau_nom = sk4 * cpsi * sphi + ck4 * spsi
		tau_den = np.sqrt((cpsi * cphi)**2 + (ck4*cpsi*sphi - sk4*spsi)**2)
		tau = np.arctan2(tau_nom, tau_den) + k[4]
		
		theta = np.pi / 2 - tau
		
		# fisheye projection with a sine (James Kumler and Martin Bauer):
		#r = k[0] * np.sin(k[1] * theta)
		# fisheye projection with a nice universal model (Donald Gennery):
		#r = k[0] * np.sin(k[1] * theta) / np.cos(np.maximum(0, k[1] * theta))
		# Rectilinear aka Perspective
		#r = k[0] * np.tan(k[1] * theta)
		# Equidistant aka equi-angular
		#r = k[0] * k[1] * theta
		# Quadratic
		#r = k[0] * theta + k[1] * theta**2
		# Cubic
		r = k[0] * theta + k[1] * theta**3
		# Fish-Eye Transform (FET, Basu and Licardie)
		#r = k[0] * np.log(1 + k[1] * np.tan(theta))
		# Equisolid
		#r = k[0] * k[1] * np.sin(theta/2)
		# Orthographic
		#r = k[0] * k[1] * np.sin(theta)
		
		crho = np.cos(- np.pi / 2 - rho)
		srho = np.sin(- np.pi / 2 - rho)
		
		a = r * crho + k[5]
		b = r * srho + k[6]
		
		return a, b
	
	def unproject(self, pos, k):
		rcrho = pos[0] - k[5]
		rsrho = pos[1] - k[6]
		
		rho = -np.pi / 2 - np.arctan2(rsrho, rcrho)
		
		r = np.sqrt(rcrho ** 2 + rsrho ** 2)
		
		# Polynomial
		theta = abs((-k[0] - np.sqrt(k[0] * k[0] - 4 * k[1] * (-r))) / (2 * k[1]))
		tau = np.pi / 2 - theta - k[4]
		
		ctau = np.cos(tau)
		stau = np.sin(tau)
		crho = np.cos(rho)
		srho = np.sin(rho)
		
		ck4 = np.cos(k[3])
		sk4 = -np.sin(k[3])
		
		# az
		phi_nom = ck4 * ctau * srho - sk4 * stau
		phi_den = ctau * crho
		phi = np.arctan2(phi_nom, phi_den) - k[2]
		
		if phi < 0:
			phi += 2 * np.pi
		
		# alt
		psi_nom = sk4 * ctau * srho + ck4 * stau
		psi_den = np.sqrt((ctau * crho)**2 + (ck4*ctau*srho - sk4*stau)**2)
		psi = np.arctan2(psi_nom, psi_den)
		
		return psi, phi

class Calibration:
	def __init__(self, location = None, catalog = None):
		self.coeff = None
		
		self.projector = Projector()
		
		if catalog is None:
			self.catalog = SkyCatalog()
		else:
			self.catalog = catalog
		
		if location is None:
			location = EarthLocation(lat=Configuration.latitude, lon=Configuration.longitude, height=Configuration.elevation)
		
		self.catalog.setLocation(location)
	
	def selectImage(self, filename):
		time = SkyCameraFile.parseTime(filename)
		self.catalog.setTime(time)
		self.catalog.calculate()
	
	def save(self, calibration_file=None):
		if calibration_file is None:
			calibration_file = Configuration.calibration_file
		
		with open(calibration_file, 'wb') as f:
			pickle.dump(self.coeff, f)
	
	def load(self, calibration_file=None):
		if calibration_file is None:
			calibration_file = Configuration.calibration_file
		
		with open(calibration_file, 'rb') as f:
			self.coeff = pickle.load(f)
	
	def project(self, pos=None, k=None):
		if pos is None:
			pos = np.array([self.catalog.alt.radian, self.catalog.az.radian]).transpose()
		
		if k is None:
			k = self.coeff
			
		return self.projector.project(pos, k)
	
	def unproject(self, pos, k=None):
		if k is None:
			k = self.coeff
			
		return self.projector.unproject(pos, k)

class Calibrator:
	def __init__(self, files, location = None, catalog = None):
		if files == None:
			files = []
		
		self.files = files
		
		if len(self.files) > 0:
			self.current_file = SkyCameraFile.uniqueName(files[0])
			self.correspondences = {self.current_file: []}
		else:
			self.current_file = None
			self.correspondences = {}
		
		self.min_max_range = 5
		self.orders = 1
		self.nonlinear = True
		self.parameter_set = 0
		self.image = None
		
		self.calibration = Calibration(location, catalog)
	
	def addImage(self, filename):
		self.files.append(filename)
		return len(self.files) - 1
	
	def selectImage(self, number, load=True):
		filename = self.files[number]
		self.current_file = SkyCameraFile.uniqueName(filename)
		
		if self.current_file not in self.correspondences:
			self.correspondences[self.current_file] = []
		
		if load:
			self.image = cv2.imread(filename)
			
		self.calibration.selectImage(filename)
		
	def findImageStar(self, x, y):
		x -= self.min_max_range
		y -= self.min_max_range
		roi = self.image[y:y + 2 * self.min_max_range + 1, x:x + 2 * self.min_max_range + 1]
		roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		_, _, _, maxLoc = cv2.minMaxLoc(roi)
		x += maxLoc[0]
		y += maxLoc[1]
		
		return x, y
	
	def getCurrentCorrespondences(self):
		return self.correspondences[self.current_file]
	
	def addCorrespondence(self, pos, altaz):
		self.correspondences[self.current_file].append(StarCorrespondence(pos, altaz))
		return len(self.correspondences[self.current_file]) - 1
	
	def removeCorrespondence(self, index):
		del(self.correspondences[self.current_file][index])
		
	def findAltAzCorrespondence(self, altaz):
		for index, correspondence in enumerate(self.correspondences[self.current_file]):
			if correspondence.altaz == altaz:
				return index
		
		return None
	
	def setCorrespondencePos(self, index, pos):
		if index is None:
			return False
		
		if self.correspondences[self.current_file][index].pos is None:
			self.correspondences[self.current_file][index].pos = pos
			return True
		
		return False
	
	def setCorrespondenceAltaz(self, index, altaz):
		if index is None:
			return False
		
		if self.correspondences[self.current_file][index].altaz is None:
			self.correspondences[self.current_file][index].altaz = altaz
			return True
		
		return False
	
	def findEmptyPos(self):
		for index, correspondence in enumerate(reversed(self.correspondences[self.current_file])):
			if correspondence.pos is None:
				return len(self.correspondences[self.current_file]) - index - 1
			
		return None

	def findEmptyAltAz(self):
		for index, correspondence in enumerate(reversed(self.correspondences[self.current_file])):
			if correspondence.altaz is None:
				return len(self.correspondences[self.current_file]) - index - 1
			
		return None
	
	def save(self, calibration_file=None, correspondence_file=None):
		self.calibration.save(calibration_file)
		
		if correspondence_file is None:
			correspondence_file = Configuration.correspondence_file
		
		with open(correspondence_file, 'wb') as f:
			pickle.dump(self.correspondences, f)
			
	def load(self, calibration_file=None, correspondence_file=None):
		self.calibration.load(calibration_file)
		
		if correspondence_file is None:
			correspondence_file = Configuration.correspondence_file
		
		with open(correspondence_file, 'rb') as f:
			self.correspondences = pickle.load(f)
			
			if self.current_file not in self.correspondences:
				self.correspondences[self.current_file] = []
				
	def resetCurrent(self):
		self.correspondences[self.current_file] = []
				
	def altazToInput(self, altaz):
		r = 1 - altaz[:, 0] / (np.pi / 2)

		shape = altaz.shape
		
		if shape[1] != 2 or len(shape) > 2:
			raise Exception('Invalid input data for transform')
		
		if self.parameter_set == 0:
			shape = (shape[0], 1 + 2 * self.orders)
			
			inpos = np.ones(shape)

			inpos[:, 0] = r * np.cos(-np.pi / 2 - altaz[:, 1])
			inpos[:, 1] = r * np.sin(-np.pi / 2 - altaz[:, 1])
			
			for i in range(1, self.orders):
				inpos[:, 2 * i + 0] = inpos[:, 0] ** (i + 1)
				inpos[:, 2 * i + 1] = inpos[:, 1] ** (i + 1)
		elif self.parameter_set == 1:
			shape = (shape[0], 1 + 4 * self.orders)
			
			inpos = np.ones(shape)

			inpos[:, 0] = r * np.cos(-np.pi / 2 - altaz[:, 1])
			inpos[:, 1] = r * np.sin(-np.pi / 2 - altaz[:, 1])
			inpos[:, 2 * (self.orders) + 0] = altaz[:, 0]
			inpos[:, 2 * (self.orders) + 1] = altaz[:, 1]
			
			for i in range(1, self.orders):
				inpos[:, 2 * i + 0] = inpos[:, 0] ** (i + 1)
				inpos[:, 2 * i + 1] = inpos[:, 1] ** (i + 1)
				inpos[:, 2 * (self.orders + i) + 0] = altaz[:, 0] ** (i + 1)
				inpos[:, 2 * (self.orders + i) + 1] = altaz[:, 1] ** (i + 1)
		else:
			shape = (shape[0], 3 + self.orders)
			
			inpos = np.ones(shape)

			inpos[:, 0] = r * np.cos(-np.pi / 2 - altaz[:, 1])
			inpos[:, 1] = r * np.sin(-np.pi / 2 - altaz[:, 1])
			
			for i in range(0, self.orders):
				inpos[:, i + 2] = r ** (i + 1)
		
		return inpos
	
	def gatherData(self):
		correspondences = []
		for _, c in self.correspondences.items():
			for correspondence in c:
				if correspondence.pos is not None and correspondence.altaz is not None:
					correspondences.append(correspondence)

		count = len(correspondences)

		altaz = np.zeros((count, 2))
		pos = np.zeros((count, 2))
		
		for index, correspondence in enumerate(correspondences):
			pos[index, :] = correspondence.pos
			altaz[index, :] = [angle.radian for angle in correspondence.altaz]
			
		return pos, altaz
	
	def calibrateExt(self):
		self.pos, self.altaz = self.gatherData()
		
		k2 = 0.5
		k1 = 1 / np.sin(k2 * np.pi / 2) * 400
		k3 = 0
		k4 = k5 = k3
		k6 = 296.03261333
		k7 = 218.56917001
		
		k0 = [k1, k2, k3, k4, k5, k6, k7]
		
		res = minimize(self.errorFunction, k0, method='nelder-mead', options={'xtol': 1e-9, 'disp': False, 'maxfev': 1e5, 'maxiter': 1e5})
		
		return res.x
	
	def errorFunction(self, k):
		a, b = self.calibration.project(self.altaz, k)
		
		x = self.pos[:, 0]
		y = self.pos[:, 1]
		
		f = np.mean((a - x)**2 + (b - y)**2)
		
		return f
		
	def lstsq(self):
		pos, altaz = self.gatherData()
		
		inpos = self.altazToInput(altaz)
		
		coeff, _, _, _ = np.linalg.lstsq(inpos, pos)
		
		return coeff
	
	def calibrate(self):
		if self.nonlinear:
			self.calibration.coeff = self.calibrateExt()
		else:
			self.calibration.coeff = self.lstsq()
		return self.calibration.coeff
	
	def transform(self, altaz):
		if not isinstance(altaz, np.ndarray):
			altaz = np.array([a.radian for a in altaz])
		
		if len(altaz.shape) == 1:
			altaz = np.array([altaz])
		
		if self.nonlinear:
			a, b = self.calibration.project(altaz, self.calibration.coeff)
			return np.column_stack((a, b))
		else:
			inpos = self.altazToInput(altaz)
			
			pos = np.matrix(inpos) * np.matrix(self.calibration.coeff)

			return pos

class CalibratorUI:
	def __init__(self):
		if len(sys.argv) < 2:
			print('Usage: calibration <directory> [<filename>]')
			print('The supplied directory should contain the calibration images.')
			sys.exit(1)
		
		size = 640
		
		self.path = sys.argv[1]
		self.image_window = 'Image Calibration'
		self.sky_window = 'Sky Calibration'
		self.tb_image_switch = 'image'
		self.tb_max_mag = 'maximum magnitude'
		self.save_file_name = 'data'
		self.selected_star = None
		self.selected_color = (0, 0, 255)
		self.marked_color = (0, 255, 0)
		self.circle_radius = 5
		self.max_mag = 4
		self.renderer = SkyRenderer(size)
		
		try:
			self.calibrator = Calibrator(SkyCameraFile.glob(self.path), EarthLocation(lat=Configuration.latitude, lon=Configuration.longitude, height=Configuration.elevation))
		except Exception as e:
			print(e.message)
			sys.exit(2)
		
		if len(sys.argv) > 2:
			self.save_file_name = sys.argv[2]
			if os.path.exists(self.save_file_name):
				self.calibrator.load(self.save_file_name)

		cv2.namedWindow(self.image_window, cv2.WINDOW_AUTOSIZE)
		cv2.namedWindow(self.sky_window, cv2.WINDOW_AUTOSIZE)
		
		self.selectImage(0)

		cv2.setMouseCallback(self.image_window, self.imageMouseCallback)
		cv2.setMouseCallback(self.sky_window, self.skyMouseCallback)
		cv2.createTrackbar(self.tb_image_switch, self.image_window, 0, len(self.calibrator.files) - 1, self.selectImage)
		cv2.createTrackbar(self.tb_max_mag, self.sky_window, self.max_mag, 6, self.setMaxMag)
	
	def selectImage(self, number):
		self.calibrator.selectImage(number)
		self.renderer.renderCatalog(self.calibrator.calibration.catalog, self.max_mag)
		self.selected_star = None
		self.render()
		
	def setMaxMag(self, mag):
		self.max_mag = mag
		self.renderer.renderCatalog(self.calibrator.calibration.catalog, self.max_mag)
		self.render()
		
	def render(self):
		image = self.calibrator.image.copy()
		sky = cv2.cvtColor(self.renderer.image.copy(), cv2.COLOR_GRAY2BGR)
		
		correspondences = self.calibrator.getCurrentCorrespondences()
		
		for correspondence in correspondences:
			if correspondence.pos is not None:
				cv2.circle(image, correspondence.pos, self.circle_radius, self.marked_color)
			if correspondence.altaz is not None:
				self.renderer.highlightStar(sky, correspondence.altaz, self.circle_radius, self.marked_color)
		
		if self.selected_star is not None:
			if correspondences[self.selected_star].pos is not None:
				cv2.circle(image, correspondences[self.selected_star].pos, self.circle_radius, self.selected_color)
			if correspondences[self.selected_star].altaz is not None:
				self.renderer.highlightStar(sky, correspondences[self.selected_star].altaz, self.circle_radius, self.selected_color)
		
		cv2.imshow(self.image_window, image)
		cv2.imshow(self.sky_window, sky)
		
	def renderCalibrationResult(self):
		num_circles = 9
		num_points = 64
		num = num_circles * num_points
		altaz = np.ones((num, 2))
		
		for c in range(num_circles):
			alt = c / 18 * np.pi
			for p in range(num_points):
				az = p / num_points * 2 * np.pi
				index = c * num_points + p
				altaz[index, 0] = alt
				altaz[index, 1] = az
		
		pos = self.calibrator.transform(altaz)
		inpos = self.renderer.altazToPos(altaz)
		
		image = self.calibrator.image.copy()
		sky = cv2.cvtColor(self.renderer.image.copy(), cv2.COLOR_GRAY2BGR)
		
		for c in range(num_circles):
			pts = np.array(pos[c * num_points:(c + 1) * num_points, :], np.int32)
			pts = pts.reshape((-1,1,2))
			cv2.polylines(image, [pts], True, (255, 0, 0))
			
			pts = np.array(inpos[c * num_points:(c + 1) * num_points, 0:2], np.int32)
			pts = pts.reshape((-1,1,2))
			cv2.polylines(sky, [pts], True, (255, 0, 0))
		
		correspondences = self.calibrator.getCurrentCorrespondences()
		
		for correspondence in correspondences:
			if correspondence.pos is not None:
				cv2.circle(image, correspondence.pos, self.circle_radius, self.selected_color)
			if correspondence.altaz is not None:
				altaz = correspondence.altaz
				
				pos = np.array(self.calibrator.transform(altaz), np.int32)[0] # np.array([np.array([a.radian for a in altaz])]) TODO
				pos = (pos[0], pos[1])

				cv2.circle(image, pos, self.circle_radius, self.marked_color)
				self.renderer.highlightStar(sky, correspondence.altaz, self.circle_radius, self.marked_color)

		cv2.imshow(self.image_window, image)
		cv2.imshow(self.sky_window, sky)
		
	def findCorrespondence(self, x, y):
		correspondences = self.calibrator.getCurrentCorrespondences()
		
		r2 = self.circle_radius * self.circle_radius
		
		for index, correspondence in enumerate(correspondences):
			if correspondence.pos is None:
				continue
			
			diff = np.subtract(correspondence.pos, (x, y))
			
			if np.dot(diff, diff) <= r2:
				return index
			
		return None
	
	def deleteSelectedStar(self):
		if self.selected_star is not None:
			self.calibrator.removeCorrespondence(self.selected_star)
			self.selected_star = None
			self.render()
		
	def imageMouseCallback(self, event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			x, y = self.calibrator.findImageStar(x, y)
			
			correspondence = self.findCorrespondence(x, y)
			
			if correspondence is None:
				if self.calibrator.setCorrespondencePos(self.selected_star, (x, y)):
					self.selected_star = self.calibrator.findEmptyPos()
				else:
					self.selected_star = self.calibrator.addCorrespondence((x, y), None)
			else:
				self.selected_star = correspondence
			
			self.render()
			
		elif event == cv2.EVENT_RBUTTONDOWN:
			self.deleteSelectedStar()

	def skyMouseCallback(self, event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			res = self.renderer.findStar(x, y, self.circle_radius)
			
			if res is None:
				return
			
			altaz = (res[0], res[1])
			correspondence = self.calibrator.findAltAzCorrespondence(altaz)
			
			if correspondence is None:
				if self.calibrator.setCorrespondenceAltaz(self.selected_star, altaz):
					self.selected_star = self.calibrator.findEmptyAltAz()
				else:
					self.selected_star = self.calibrator.addCorrespondence(None, altaz)
			else:
				self.selected_star = correspondence
			
			self.render()
		elif event == cv2.EVENT_RBUTTONDOWN:
			self.deleteSelectedStar()

	def run(self):
		quit = False

		while not quit:
			k = cv2.waitKey(0) & 0xFF
			if k == 27 or k == ord('q'): # ESC
				quit = True
				
			elif k == ord('c'):
				coeff = self.calibrator.calibrate()

				error = self.calibrator.errorFunction(coeff)
				
				print('Calibration result:', error)
				print(coeff)
				
				self.renderCalibrationResult()
				
			elif k == ord('d'):
				self.renderCalibrationResult()
				
			elif k == ord('s'):
				self.calibrator.save(self.save_file_name)
				print('Saved')
				
			elif k == ord('l'):
				self.calibrator.load(self.save_file_name)
				self.selected_star = None
				self.render()
				
			elif k == ord('r'):
				self.calibrator.resetCurrent()
				self.render()

		cv2.destroyAllWindows()

if __name__ == '__main__':
	ui = CalibratorUI()
	ui.run()

#__import__("code").interact(local=locals())
