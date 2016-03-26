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

from astropy import units as u
import sys

class Configuration:
	## General Settings
	
	# Observer Position
	latitude = 47.06713 * u.deg
	longitude = 15.49343 * u.deg
	elevation = 493 * u.m
	
	# Angles for sun and moon determination
	night_angle = -6 * u.deg
	day_angle = 6 * u.deg
	moon_up_angle = 0 * u.deg
	
	# Logging
	logging = False
	log_file = 'log.txt'
	
	## Recording and Camera Settings
	
	# Recording Settings
	day_averaging_frames = 100
	night_averaging_frames = 1
	default_storage_path = 'frames'
	day_time_between_frames = 20 # seconds
	night_time_between_frames = 0#1#0 # seconds
	frame_count = -1 # <= 0 means infinite
	show_recorded_frames = True
	store_in_subdirectory = True
	
	# Settings for dynamic night frame averaging
	dnfa_enabled = True
	dnfa_window_size = 150
	dnfa_min_med_diff_factor = 0.2
	dnfa_min_diff_value = 0.3
	dnfa_min_frames = 50
	dnfa_max_frames = 200
	
	# Camera Settings
	control_settings = True
	
	serial_port = 7
	time_between_commands = 0.2 # seconds
	verbose_commands = True
	
	# these settings should be set and are the same for day and night: "BLC0", "FLC0", "PRAG", "CMDA", "GATB", "ENMI"
	day_settings = ["SSH0", "SSX0", "SAES", "AGC0"]
	night_settings = ["SSH1", "SSX7", "SALC", "ALC0", "AGC1"]
	# shooting star settings?
	#night_settings = ["SSH1", "SSX4", "SALC", "ALC0", "AGC1"]
	
	## Day time Algorithm Settings
	rb_difference_threshold = 6
	rb_ratio_threshold = 0.9 # 0.6
	br_ratio_threshold = 1 # 1.3
	nbr_threshold = 0.25
	adaptive_threshold = -10
	adaptive_block_size = 127
	background_rect_size = 99
	mc_rb_threshold = 20
	mc_bg_threshold = 20
	mc_b_threshold = 250
	sp_num_superpixels = 100
	sp_prior = 2
	sp_num_levels = 4
	sp_num_histogram_bins = 5
	sp_num_iterations = 4
	sp_kernel_size = 5
	morphology_kernel_size = 3
	morphology_iterations = 2
	
	## Night time Algorithm Settings
	
	# Configuration Files
	calibration_file = "calibration.dat"
	correspondence_file = "correspondences.dat"
	configuration_file = "configuration.dat"
	star_catalog_file = "catalog"
	mask_file = "mask.png"
	
	# Algorithm Settings
	min_alt = 10
	alt_step = 20
	az_step = 60
	max_mag = 2.5
	
	# Star Finder
	gaussian_roi_size = 5
	gaussian_threshold = 0.2
	gaussian_kernel_size = 33
	
	candidate_radius = 5
	
	gftt_max_corners = 600
	gftt_quality_level = 0.002
	gftt_min_distance = 5
	
	surf_threshold = 1
	
	log_max_rect_size = 10
	log_block_size = 9
	log_threshold = 0.53
	log_kernel_size = 5
	
	## Difference Settings
	
	# Difference View Settings
	difference_detection_window_size = 10
	
# here is the configuration for my development machine which runs linux, not windows
if sys.platform == 'linux':
	# Camera Settings
	Configuration.serial_port = '/dev/ttyS4'
