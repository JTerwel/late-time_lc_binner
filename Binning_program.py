'''
A program to stack and display late-time observations of transients, observed with ZTF

Author: Jacco Terwel
Date: 26-01-24

Added:
	- Updated comments
	- mkdir now also creates missing parent directories
	- It is now possible to specify the maximum amount of cpus to use for multiprocessing
	- Added a new mode: pre-ZTF
		* The initial transient is assumed to have faded beyond the detection limit before the lc starts
		* Choose which observations are used for the baseline correction
		* Run different baseline regions consecutively
		* Choose first/last N days/points or specify an mjd range
		* Choose a single or multiple regions
		* All points not in the baseline region(s) are used for binning
		* No tail fits necessary (disabled in this mode), meaning any type of transient can be used
	- Added some extra functions and extra/rewritten lines to existing functions to make the new mode work
'''

#Imports & global constants
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from numpy import nan, inf
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
from tqdm import tqdm
from lmfit import Model, Parameters
import traceback
import sys
import ast

def main():
	'''
	Main function of the program.

	Make a list of locations to find the required data for each object, set
	the location where all results will be saved (separate folders for each
	object), and control the progress bar.

	The way this script is written, it assumes to be given 3 commandline variables:
	- lc loc: The location of the lcs to bin
	- saveloc: The location to store the output
	- config: The config file
	'''
	if len(sys.argv) != 4:
		print('ERROR: Wrong amount of arguments, run the program by typing the following:')
		print(f'python3 Binning_program.py <lc loc> <save loc> <config file>\n These were given: {sys.argv}')
		return
	#Load the config file
	config = pd.read_csv(sys.argv[3], header=None, index_col=0, delim_whitespace=True, comment='#')

	#Set location where the results will be saved
	print('\nStarting the program\nCollecting objects')
	saveloc = Path(sys.argv[2])
	saveloc.mkdir(exist_ok=True, parents=True)

	#Set mode (type of input) & Set the location of the object files and list them if needed
	modes = ['ZTFDR2', 'simulations', 'fpbot', 'real_time', 'pre-ZTF']
	mode = config.loc['mode'][1]
	if ((mode == modes[0]) | (mode == modes[2]) | (mode == modes[3]) | (mode == modes[4])): #Files are csv files in a folder
		datafiles = list(Path(sys.argv[1]).rglob('*.csv'))
	elif mode == modes[1]: #No files, instead a list of simsurvey lcs that need conversion
		import simsurvey
		datafiles = simsurvey.LightcurveCollection(load=Path(sys.argv[1])/'lcs.pkl')
	else:
		print(f'Error: mode {mode} not recognised, please use one from the following list:')
		print(modes)
		print('Fatal error: cannot run program, stopping now')
		return

	#Load the reference image data
	refmags = pd.read_csv(config.loc['refmags_loc'][1],
		usecols=['field', 'fid', 'rcid', 'maglimit', 'startmjd', 'endmjd'])

	#Load camera ZPs
	read_opts = {'delim_whitespace': True, 'names': ['g', 'r', 'i'], 'comment': '#'}
	zp_rcid = pd.read_csv(config.loc['zp_rcid_loc'][1], **read_opts)

	#Set free parameters & make the arg_list
	late_time = ast.literal_eval(config.loc['late_time'][1])
	cuts = ast.literal_eval(config.loc['cuts'][1])
	base_gap = ast.literal_eval(config.loc['base_gap'][1])
	binsizes = ast.literal_eval(config.loc['binsizes'][1])
	phases = ast.literal_eval(config.loc['phases'][1])
	method = ast.literal_eval(config.loc['method'][1])
	earliest_tail_fit_start = ast.literal_eval(config.loc['earliest_tail_fit_start'][1])
	verdict_sigma = ast.literal_eval(config.loc['verdict_sigma'][1])
	tail_fit_chi2dof_tresh = ast.literal_eval(config.loc['tail_fit_chi2dof_tresh'][1])
	min_sep = ast.literal_eval(config.loc['min_sep'][1])
	min_successes = ast.literal_eval(config.loc['min_successes'][1])
	save_lc_and_bins = ast.literal_eval(config.loc['save_lc_and_bins'][1])
	note_missing_host = ast.literal_eval(config.loc['note_missing_host'][1])

	if ((mode == modes[0]) | (mode == modes[2])):
		args = [[f, refmags, saveloc, late_time, zp_rcid, cuts, base_gap, mode, binsizes, phases,
				 method, earliest_tail_fit_start, verdict_sigma, tail_fit_chi2dof_tresh,
				 save_lc_and_bins] for f in datafiles]
		nr_files = len(datafiles)
	elif mode == modes[1]:
		fields = pd.read_fwf(config.loc['fields_list_loc'][1], header=0)
		args = [[_, refmags, saveloc, late_time, zp_rcid, cuts, base_gap, mode, binsizes, phases,
				method, earliest_tail_fit_start, verdict_sigma, tail_fit_chi2dof_tresh,
				save_lc_and_bins, [datafiles[_], fields]] for _ in range(len(datafiles.lcs))]
		nr_files = len(datafiles.lcs)
	elif mode == modes[3]:
		args = [[f, refmags, saveloc, late_time, zp_rcid, cuts, base_gap, mode, binsizes, phases,
				 method, earliest_tail_fit_start, verdict_sigma, tail_fit_chi2dof_tresh,
				 save_lc_and_bins] for f in datafiles]
		nr_files = len(datafiles)
	elif mode == modes[4]:
		#Make a list of strings containing the baselines, args will be made later
		baselines = pd.read_csv(config.loc['baselines_loc'][1], header=0).date_range.values
		nr_files = len(datafiles)
	else:
		print(f'Error, {mode} not recognised when building the args list, stopping the program')
		return

	#Save all settings
	if mode != modes[3]:
		data_settings = {'cuts': cuts}
		data_settings.update({'late_time':late_time, 'binsizes':binsizes, 'phases':phases,
							  'method':method, 'min_sep':min_sep, 'min_successes':min_successes})
		settings = pd.Series(data_settings)
		settings.to_csv(saveloc / 'settings.csv')

	print('Initialization complete, starting checking each object')
	#use each cpu core separately & keep track of progress
	max_cpu = ast.literal_eval(config.loc['max_cpu'][1])
	pool = mp.Pool(min(int(max_cpu), mp.cpu_count()) if max_cpu is not None else mp.cpu_count())

	if mode == modes[4]:
		for _ in baselines:
			print(f'Running the binning program while using the following baseline region: {_}')
			args = [[f, refmags, saveloc, late_time, zp_rcid, cuts, base_gap, mode, binsizes, phases,
					 method, earliest_tail_fit_start, verdict_sigma, tail_fit_chi2dof_tresh,
					 save_lc_and_bins, _] for f in datafiles]
			list(tqdm(pool.imap_unordered(check_object, args), total=nr_files))
	else:
		list(tqdm(pool.imap_unordered(check_object, args), total=nr_files))
	pool.close()
	pool.join()
	print('All objects binned & evaluated, putting objects through the final filter')
	if mode == modes[4]:
		all_obs = [f for f in saveloc.rglob('*/*') if f.is_dir()]
	else:
		all_obs = [f for f in saveloc.rglob('*') if f.is_dir()]
	#Set host data locations
	loc_list = config.loc['loc_list_loc'][1]
	dat_list = config.loc['dat_list_loc'][1]
	if ((mode == modes[1]) | (mode == modes[4])):
		host_dat = [None]*len(all_obs)
	else:
		host_dat = get_host_data([f.name for f in saveloc.rglob('*') if f.is_dir()], loc_list, dat_list)
	#Give the final verdicts
	final_verdicts = give_final_verdicts(mode, all_obs, host_dat, min_sep, min_successes, note_missing_host)
	final_verdicts.to_csv(saveloc / 'final_verdicts.csv', index=False)
	print('Program finished\n')
	return

#*---------*
#| Classes |
#*---------*

class ztf_object:
	'''
	Class holding all information of an ZTF object + additional useful data

	Attributes:
	- name (str): ZTF name
	- loc (Path): lc location
	- refmags (DataFrame): reference image magnitude limits
	- saveloc (Path): location to save results
	- late_time (float): start of late-time observations (days after peak)
	- zp_rcid (DataFrame): camera zeropoints
	- cuts (dict): used cuts for improved data_quality
	- base_gap (float): min time (days) between observation & peak to be considered for the baseline correction
	- mode (string): Input type to expect (Each has its own load_source function)
	- binsizes (list): binsizes to use when binning
	- phases (list): offsets to the 1st bin to use when binning
	- method (int): method used to choose bin positioning (see bin_late_time)
	- earliest_tail_fit_start (float): earliest observations relative to peak to consider when fitting the tail
	- verdict_sigma (float): significance level for a bin to be considered a detection
	- tail_fit_chi2dof_tresh: chi2dof treshold for a successful tail fit
	- peak_mjd (float): found mjd of the SN peak
	- peak_mag (float): found mag of the SN peak
	- final_mjd (dict): final observation mjd for each band
	- text (string): Notes on this object, will be saved in a .txt file
	- binlist (list): list of binning attempts, one for each filter, binsize, phase cobination
	- verdictlist (list): list of verdicts for each binning attempt in binlist
	- imp_df (DataFrame): df containing other values to save (peak_date, baseline_corrections)
	- extra_args (list): list of extra arguments needed for loading the specific input type
	'''

	def __init__(self, args):
		#Unpack args into easier to use names
		self.mode = args [7]
		#Naming scheme is chosen based on mode, if not recognised go with the default option
		if self.mode == 'ZTFDR2':
			self.name = args[0].name.rsplit('_L',1)[0]
			if (('ZTF' not in self.name) & ('ztf' not in self.name)):
				self.name = args[0].name[:-4]
		elif self.mode == 'simulations':
			self.name = f'lc_{args[0]}'
		elif self.mode == 'fpbot':
			self.name = args[0].name.rsplit('_SNT',1)[0]
		elif ((self.mode == 'real_time') | (self.mode == 'pre-ZTF')):
			self.name = Path(args[0]).stem
		else: #It should not be possible to get here, but just in case
			self.name = args[0]
		self.loc = args[0]
		self.refmags = args[1]
		self.saveloc = args[2] / self.name
		self.late_time = args[3]
		self.zp_rcid = args[4]
		self.cuts = args[5]
		self.base_gap = args[6]
		self.binsizes = args[8]
		self.phases = args[9]
		self.method = args[10]
		self.earliest_tail_fit_start = args[11]
		self.verdict_sigma = args[12]
		self.tail_fit_chi2dof_tresh = args[13]
		self.save_lc_and_bins = args[14]
		self.text = 'Notes on ' + self.name + ':\n\n'
		#Initialize the lists that will store the bins & verdicts
		self.binlist = []
		self.verdictlist = []
		self.imp_df = pd.DataFrame(columns=['name', 'val'])
		#Make the directory to save things in
		self.saveloc.mkdir(exist_ok=True, parents=True)
		#Load the lc
		if self.mode == 'ZTFDR2':
			self.load_source_dr2()
		elif self.mode == 'simulations':
			self.extra_args = args[15]
			self.load_source_sims()
		elif ((self.mode == 'fpbot') | (self.mode == 'real_time')):
			self.load_source_fpbot()
		elif self.mode == 'pre-ZTF':
			self.baseline_request = args[15]
			self.load_source_pre_ZTF()
		#Note the final mjd for each band
		if len(self.lc) != 0:
			self.note_final_mjd()
		return

	def load_source_dr2(self):
		#Load the lc assuming a DR2 style dataframe
		cols = ['mjd', 'filter', 'flux', 'flux_err', 'ZP', 'flag', 'mag', 'mag_err', 'field_id', 'rcid', 'flux_offset', 'err_scale']
		try:
			self.lc = pd.read_csv(self.loc, usecols = cols, comment='#', delim_whitespace=True)
		except: #In case something goes wrong when reading in
			self.text += f'Could not read in lc for {self.name}\n'+ traceback.format_exc() + '\n'
			self.peak_mjd = 0
			self.lc = pd.DataFrame(columns=cols)
			return
		#Rename problematic columns
		self.lc.rename(columns={'mjd':'obsmjd', 'filter':'obs_filter', 'flux':'Fratio', 'flux_err':'Fratio_err', 'field_id':'fieldid'}, inplace=True)
		#Remove rows with rcid = NaN
		self.lc = self.lc[~self.lc.rcid.isnull().values].reset_index(drop=True)
		#Cloudy & cuts are in flag, Undo idr baseline correction & add refmags
		self.undo_correction()
		self.lc = self.lc[self.lc.flag&sum(self.cuts)==0]
		self.match_ref_im()
		#Find the peak of the SN
		self.find_peak_date()
		#Save this lc before bad points are removed if wanted
		nr_points = [len(self.lc[self.lc.obs_filter.str.contains('g')]),
					 len(self.lc[self.lc.obs_filter.str.contains('r')]),
					 len(self.lc[self.lc.obs_filter.str.contains('i')])]
		if self.save_lc_and_bins:
			self.lc.to_csv(self.saveloc/'lc_orig.csv')
		self.imp_df = self.imp_df.append({'name':'peak_mjd_before_baseline', 'val':self.peak_mjd},
										 ignore_index=True)
		#apply baseline corrections
		self.correct_baseline()
		#Save the baseline corrected lc containing only useful points if wanted
		if self.save_lc_and_bins:
			self.lc.to_csv(self.saveloc/'lc_cor.csv')
		if self.lc.empty:
			self.text += '-----\nNo good datapoints left, no binning can be performed\n'
		#Recheck the peak of the SN in case the relevant point was removed
		tmp1 = self.peak_mjd
		tmp2 = self.peak_mag
		self.find_peak_date()
		self.imp_df = self.imp_df.append({'name':'peak_mjd_after_baseline', 'val':self.peak_mjd},
										 ignore_index=True)
		self.imp_df = self.imp_df.append({'name':'removed_g',
										  'val':nr_points[0]-len(self.lc[self.lc.obs_filter.str.contains('g')])},
										 ignore_index=True)
		self.imp_df = self.imp_df.append({'name':'removed_r',
										  'val':nr_points[1]-len(self.lc[self.lc.obs_filter.str.contains('r')])},
										 ignore_index=True)
		self.imp_df = self.imp_df.append({'name':'removed_i',
										  'val':nr_points[2]-len(self.lc[self.lc.obs_filter.str.contains('i')])},
										 ignore_index=True)
		if ((self.peak_mjd != tmp1) | (self.peak_mag != tmp2)):
			self.text += f'NOTE: Peak has changed after baseline corrections by {self.peak_mjd-tmp1} days and {self.peak_mag-tmp2} mags\n'
		self.check_refmjd()
		return

	def load_source_sims(self):
		#Read in simulated lc & convert to fpbot format
		self.lc = make_fpbot_df(self.extra_args[0], self.extra_args[1])
		# Update the filter entries for consistency.
		self.lc.loc[self.lc.obs_filter == 'ZTF g', 'obs_filter'] = 'ztfg'
		self.lc.loc[self.lc.obs_filter == 'ZTF r', 'obs_filter'] = 'ztfr'
		self.lc.loc[self.lc.obs_filter == 'ZTF i', 'obs_filter'] = 'ztfi'
		self.lc.loc[self.lc.obs_filter == 'ZTF_g', 'obs_filter'] = 'ztfg'
		self.lc.loc[self.lc.obs_filter == 'ZTF_r', 'obs_filter'] = 'ztfr'
		self.lc.loc[self.lc.obs_filter == 'ZTF_i', 'obs_filter'] = 'ztfi'
		#Remove rows with rcid = NaN
		self.lc = self.lc[~self.lc.rcid.isnull().values].reset_index(drop=True)
		# Remove duplicate observations: (uses field, rcid, filter and mjd)
		self.lc = self.lc.drop_duplicates(subset=['fieldid','obs_filter','rcid','obsmjd'], keep='last')
		self.lc = self.lc.reset_index(drop=True)
		#Add flags and transform to dr2-like df, add cloudy, cuts, & refmags, Remove points where the SN is not on the chip
		#This should remove nothing important
		self.calc_cloudy()
		self.apply_cuts()
		self.lc = self.lc[self.lc.flag&sum(self.cuts)==0]
		if len(self.lc.loc[(self.lc.target_x<0) | (self.lc.target_y<0)])>0:
			self.lc = self.lc.loc[(self.lc.target_x>=0) & (self.lc.target_y>=0)]
		self.match_ref_im()
		#zp already at 30, no need to renormalise
		#Find the peak of the SN
		self.find_peak_date()
		#Save this lc before bad points are removed if wanted
		nr_points = [len(self.lc[self.lc.obs_filter.str.contains('g')]),
					 len(self.lc[self.lc.obs_filter.str.contains('r')]),
					 len(self.lc[self.lc.obs_filter.str.contains('i')])]
		if self.save_lc_and_bins:
			self.lc.to_csv(self.saveloc/'lc_orig.csv')
		self.imp_df = self.imp_df.append({'name':'peak_mjd_before_baseline', 'val':self.peak_mjd},
										 ignore_index=True)
		#apply baseline corrections
		self.correct_baseline()
		#Save the baseline corrected lc containing only useful points if wanted
		if self.save_lc_and_bins:
			self.lc.to_csv(self.saveloc/'lc_cor.csv')
		if self.lc.empty:
			self.text += '-----\nNo good datapoints left, no binning can be performed\n'
		#Recheck the peak of the SN in case the relevant point was removed
		tmp1 = self.peak_mjd
		tmp2 = self.peak_mag
		self.find_peak_date()
		self.imp_df = self.imp_df.append({'name':'peak_mjd_after_baseline', 'val':self.peak_mjd},
										 ignore_index=True)
		self.imp_df = self.imp_df.append({'name':'removed_g',
										  'val':nr_points[0]-len(self.lc[self.lc.obs_filter.str.contains('g')])},
										 ignore_index=True)
		self.imp_df = self.imp_df.append({'name':'removed_r',
										  'val':nr_points[1]-len(self.lc[self.lc.obs_filter.str.contains('r')])},
										 ignore_index=True)
		self.imp_df = self.imp_df.append({'name':'removed_i',
										  'val':nr_points[2]-len(self.lc[self.lc.obs_filter.str.contains('i')])},
										 ignore_index=True)
		if ((self.peak_mjd != tmp1) | (self.peak_mag != tmp2)):
			self.text += f'NOTE: Peak has changed after baseline corrections by {self.peak_mjd-tmp1} days and {self.peak_mag-tmp2} mags\n'
		self.check_refmjd()
		return

	def load_source_fpbot(self):
		#Load the lc assuming a raw fpbot style dataframe
		if self.mode == 'real_time':
			#The real-time version of fpbot stops before the plot command, which add the last 7 columns
			#Therefore we'll need to make them here if they're needed
			cols = ['obsmjd', 'filter', 'ampl',
					'ampl.err', 'chi2dof', 'seeing', 'magzp', 'magzpunc', 'magzprms', 'airmass', 'nmatches',
					'rcid', 'fieldid', 'infobits', 'filename', 'exptime', 'sigma', 'sigma.err', 'target_x',
					'target_y', 'maglim']
			try:
				self.lc = pd.read_csv(self.loc, usecols = cols, comment='#')
			except: #In case something goes wrong when reading in
				self.text += f'Could not read in lc for {self.name}\n'+ traceback.format_exc() + '\n'
				self.peak_mjd = 0
				self.lc = pd.DataFrame(columns=cols)
				return
			#Need to make Fratio, Fratio.err, mag, mag_err, upper_limit, and need F0 and F0_err for that
			F0 = 10 ** (self.lc.magzp / 2.5)
			F0_err = F0 / 2.5 * np.log(10) * self.lc.magzpunc
			self.lc["Fratio"] = self.lc.ampl / F0
			self.lc["Fratio.err"] = np.sqrt((self.lc["ampl.err"]/F0) ** 2+(self.lc.ampl*F0_err/F0**2)**2)
			mags = []
			mags_unc = []
			upper_limits = []
			Fratios = np.asarray(self.lc["Fratio"].values)
			Fratios_unc = np.asarray(self.lc["Fratio.err"].values)
			maglims = np.asarray(self.lc["maglim"].values)
			for i, Fratio in enumerate(Fratios):
				Fratio_unc = Fratios_unc[i]
				if Fratio > (Fratio_unc * 5): #Fixed snt to 5
					upper_limit = np.nan
					mag = -2.5 * np.log10(Fratio)
					mag_unc = 2.5 / np.log(10) * Fratio_unc / Fratio
				else:
					upper_limit = maglims[i]
					mag = 99
					mag_unc = 99
				upper_limits.append(upper_limit)
				mags.append(mag)
				mags_unc.append(mag_unc)
			self.lc["upper_limit"] = upper_limits
			self.lc["mag"] = mags
			self.lc["mag_err"] = mags_unc
		else: #Normal fpbot format coming from the Slackbot
			cols = ['obsmjd', 'filter', 'Fratio', 'Fratio.err', 'mag', 'mag_err', 'upper_limit', 'ampl',
					'ampl.err', 'chi2dof', 'seeing', 'magzp', 'magzpunc', 'magzprms', 'airmass', 'nmatches',
					'rcid', 'fieldid', 'infobits', 'filename', 'exptime', 'sigma', 'sigma.err', 'target_x',
					'target_y']
			try:
				self.lc = pd.read_csv(self.loc, usecols = cols, comment='#')
			except: #In case something goes wrong when reading in
				self.text += f'Could not read in lc for {self.name}\n'+ traceback.format_exc() + '\n'
				self.peak_mjd = 0
				self.lc = pd.DataFrame(columns=cols)
				return
		#Rename problematic columns
		self.lc.rename(columns={'filter':'obs_filter', 'Fratio.err':'Fratio_err',
						   		'ampl.err':'ampl_err', 'sigma.err':'sigma_err'}, inplace=True)
		if len(self.lc) == 0: #lc turns out to be empty --> No use in trying to do any of the following steps
			self.peak_mjd = 0
			return
		# Update the filter entries for consistency.
		self.lc.loc[self.lc.obs_filter == 'ZTF g', 'obs_filter'] = 'ztfg'
		self.lc.loc[self.lc.obs_filter == 'ZTF r', 'obs_filter'] = 'ztfr'
		self.lc.loc[self.lc.obs_filter == 'ZTF i', 'obs_filter'] = 'ztfi'
		self.lc.loc[self.lc.obs_filter == 'ZTF_g', 'obs_filter'] = 'ztfg'
		self.lc.loc[self.lc.obs_filter == 'ZTF_r', 'obs_filter'] = 'ztfr'
		self.lc.loc[self.lc.obs_filter == 'ZTF_i', 'obs_filter'] = 'ztfi'
		#Remove rows with rcid = NaN
		self.lc = self.lc[~self.lc.rcid.isnull().values].reset_index(drop=True)
		# Remove duplicate observations: (uses field, rcid, filter and mjd)
		self.lc = self.lc.drop_duplicates(subset=['fieldid','obs_filter','rcid','obsmjd'], keep='last')
		self.lc = self.lc.reset_index(drop=True)
		#Add flags and transform to dr2-like df, add cloudy, cuts, & refmags, Remove points where the SN is not on the chip
		self.calc_cloudy()
		self.apply_cuts()
		self.lc = self.lc[self.lc.flag&sum(self.cuts)==0]
		if len(self.lc.loc[(self.lc.target_x<0) | (self.lc.target_y<0)])>0:
			self.lc = self.lc.loc[(self.lc.target_x>=0) & (self.lc.target_y>=0)]
		#Match reference images
		self.match_ref_im()
		#Put everything to zp 30
		self.renormalise_fpbot_lc()
		#Find the peak of the SN
		self.find_peak_date()
		#Save this lc before bad points are removed if wanted
		nr_points = [len(self.lc[self.lc.obs_filter.str.contains('g')]),
					 len(self.lc[self.lc.obs_filter.str.contains('r')]),
					 len(self.lc[self.lc.obs_filter.str.contains('i')])]
		if self.save_lc_and_bins:
			self.lc.to_csv(self.saveloc/'lc_orig.csv')
		self.imp_df = self.imp_df.append({'name':'peak_mjd_before_baseline', 'val':self.peak_mjd},
										 ignore_index=True)
		#apply baseline corrections
		self.correct_baseline()
		#Save the baseline corrected lc containing only useful points if wanted
		if self.save_lc_and_bins:
			self.lc.to_csv(self.saveloc/'lc_cor.csv')
		if self.lc.empty:
			self.text += '-----\nNo good datapoints left, no binning can be performed\n'
		#Recheck the peak of the SN in case the relevant point was removed
		tmp1 = self.peak_mjd
		tmp2 = self.peak_mag
		self.find_peak_date()
		self.imp_df = self.imp_df.append({'name':'peak_mjd_after_baseline', 'val':self.peak_mjd},
										 ignore_index=True)
		self.imp_df = self.imp_df.append({'name':'removed_g',
										  'val':nr_points[0]-len(self.lc[self.lc.obs_filter.str.contains('g')])},
										 ignore_index=True)
		self.imp_df = self.imp_df.append({'name':'removed_r',
										  'val':nr_points[1]-len(self.lc[self.lc.obs_filter.str.contains('r')])},
										 ignore_index=True)
		self.imp_df = self.imp_df.append({'name':'removed_i',
										  'val':nr_points[2]-len(self.lc[self.lc.obs_filter.str.contains('i')])},
										 ignore_index=True)
		if ((self.peak_mjd != tmp1) | (self.peak_mag != tmp2)):
			self.text += f'NOTE: Peak has changed after baseline corrections by {self.peak_mjd-tmp1} days and {self.peak_mag-tmp2} mags\n'
		self.check_refmjd()
		return

	def load_source_pre_ZTF(self):
		#Check if the pre-baseline correction lc exists. If so load it, else make it
		if (self.saveloc / 'lc_orig.csv').is_file():
			try:
				self.lc = pd.read_csv(self.saveloc / 'lc_orig.csv', header=0)
				found_orig = True
			except:
				found_orig = False
		else:
			found_orig = False
		if not found_orig:
			#The fpbot stopped before the plot command, which add the last 7 columns
			#Therefore we'll need to make them here if they're needed
			cols = ['obsmjd', 'filter', 'ampl',
					'ampl.err', 'chi2dof', 'seeing', 'magzp', 'magzpunc', 'magzprms', 'airmass', 'nmatches',
					'rcid', 'fieldid', 'infobits', 'filename', 'exptime', 'sigma', 'sigma.err', 'target_x',
					'target_y', 'maglim']
			try:
				self.lc = pd.read_csv(self.loc, usecols = cols, comment='#')
			except: #In case something goes wrong when reading in
				self.text += f'Could not read in lc for {self.name}\n'+ traceback.format_exc() + '\n'
				self.peak_mjd = 0
				self.lc = pd.DataFrame(columns=cols)
				return
			#Need to make Fratio, Fratio_err, mag, mag_err, upper_limit, and need F0 and F0_err for that
			F0 = 10 ** (self.lc.magzp / 2.5)
			F0_err = F0 / 2.5 * np.log(10) * self.lc.magzpunc
			self.lc["Fratio"] = self.lc.ampl / F0
			self.lc["Fratio_err"] = np.sqrt((self.lc["ampl.err"]/F0) ** 2+(self.lc.ampl*F0_err/F0**2)**2)
			mags = []
			mags_unc = []
			upper_limits = []
			Fratios = np.asarray(self.lc["Fratio"].values)
			Fratios_unc = np.asarray(self.lc["Fratio_err"].values)
			maglims = np.asarray(self.lc["maglim"].values)
			for i, Fratio in enumerate(Fratios):
				Fratio_unc = Fratios_unc[i]
				if Fratio > (Fratio_unc * 5): #Fixed snt to 5
					upper_limit = np.nan
					mag = -2.5 * np.log10(Fratio)
					mag_unc = 2.5 / np.log(10) * Fratio_unc / Fratio
				else:
					upper_limit = maglims[i]
					mag = 99
					mag_unc = 99
				upper_limits.append(upper_limit)
				mags.append(mag)
				mags_unc.append(mag_unc)
			self.lc["upper_limit"] = upper_limits
			self.lc["mag"] = mags
			self.lc["mag_err"] = mags_unc
			#Rename problematic columns
			self.lc.rename(columns={'filter':'obs_filter', 'ampl.err':'ampl_err', 'sigma.err':'sigma_err'},
						   inplace=True)
			if len(self.lc) == 0: #lc turns out to be empty --> No use in trying to do any of the following steps
				self.peak_mjd = 0
				return
			# Update the filter entries for consistency.
			self.lc.loc[self.lc.obs_filter == 'ZTF g', 'obs_filter'] = 'ztfg'
			self.lc.loc[self.lc.obs_filter == 'ZTF r', 'obs_filter'] = 'ztfr'
			self.lc.loc[self.lc.obs_filter == 'ZTF i', 'obs_filter'] = 'ztfi'
			self.lc.loc[self.lc.obs_filter == 'ZTF_g', 'obs_filter'] = 'ztfg'
			self.lc.loc[self.lc.obs_filter == 'ZTF_r', 'obs_filter'] = 'ztfr'
			self.lc.loc[self.lc.obs_filter == 'ZTF_i', 'obs_filter'] = 'ztfi'
			#Remove rows with rcid = NaN
			self.lc = self.lc[~self.lc.rcid.isnull().values].reset_index(drop=True)
			# Remove duplicate observations: (uses field, rcid, filter and mjd)
			self.lc = self.lc.drop_duplicates(subset=['fieldid','obs_filter','rcid','obsmjd'], keep='last')
			self.lc = self.lc.reset_index(drop=True)
			#Add flags and transform to dr2-like df, add cloudy, cuts, & refmags, Remove points where the SN is not on the chip
			self.calc_cloudy()
			self.apply_cuts()
			self.lc = self.lc[self.lc.flag&sum(self.cuts)==0]
			if len(self.lc.loc[(self.lc.target_x<0) | (self.lc.target_y<0)])>0:
				self.lc = self.lc.loc[(self.lc.target_x>=0) & (self.lc.target_y>=0)]
			self.lc.reset_index(drop=True, inplace=True)
			#Match reference images
			self.match_ref_im()
			#Put everything to zp 30
			self.renormalise_fpbot_lc()
			#Save this lc before bad points are removed if wanted
			nr_points = [len(self.lc[self.lc.obs_filter.str.contains('g')]),
						 len(self.lc[self.lc.obs_filter.str.contains('r')]),
						 len(self.lc[self.lc.obs_filter.str.contains('i')])]
			self.lc.to_csv(self.saveloc/'lc_orig.csv')
			#Save the notes on making the lc up to applying the baseline correction & reset it
			ftext = open(self.saveloc/'notes.txt', 'w')
			ftext.write(self.text)
			ftext.close()
			self.text = f'Notes on {self.name}:\n\n'
		#Set the range to be used for the baseline correction
		self.select_baseline()
		if len(self.baselines) == 0:
			text += 'No proper baseline region found\nReturning an empty DataFrame'
			self.lc = pd.DataFrame(columns=cols)
			return
		#From here on, everything will be saved in the baseline specific subfolder, so update saveloc and make directory
		#folder naming convention = b<# baseline regions>_<1st baseline start mjd>_<1st baseline end mjd>
		self.saveloc = self.saveloc / f'b{len(self.baselines)}_{self.baselines[0][0]:.0f}_{self.baselines[0][1]:.0f}'
		self.saveloc.mkdir(exist_ok=True, parents=True)
		self.correct_baseline()
		self.lc.reset_index(drop=True, inplace=True)
		#Drop the points that were used for the baseline correction as they can't be used in the binning
		to_bin = [True]*len(self.lc)
		for r in self.baselines:
			for _ in self.lc.index:
				if ((self.lc.loc[_, 'obsmjd'] >= r[0]) & (self.lc.loc[_, 'obsmjd'] <= r[1])):
					to_bin[_] = False
		self.lc = self.lc[to_bin]
		self.lc.reset_index(drop=True, inplace=True)
		#Save the baseline corrected lc if wanted
		if self.save_lc_and_bins:
			self.lc.to_csv(self.saveloc/'lc_cor.csv')
		if self.lc.empty:
			self.text += '-----\nNo good datapoints left, no binning can be performed\n'
		#Set peak_mjd & late_time to dummy values to ensure the program works and bins all remaining points
		self.peak_mjd = 1
		self.late_time = 100
		#Note when the references were made
		if not self.lc.empty:
			self.check_refmjd()
		return

	def note_final_mjd(self):
		'''
		Note down the final mjd in each band
		'''
		self.final_mjd = {}
		for band in ['g', 'r', 'i']:
			if len(self.lc[self.lc.obs_filter.str.contains(band)]) > 0:
				self.final_mjd[band] = self.lc[self.lc.obs_filter.str.contains(band)].obsmjd.max()
		return

	def undo_correction(self):
		'''
		Undo the corrections applied to the flux/Fratio & flux_err/Fratio_err in preparation to use my own
		'''
		self.lc.Fratio = self.lc.Fratio + self.lc.flux_offset
		self.lc.Fratio_err = self.lc.Fratio_err / self.lc.err_scale
		#update mag values & add upper limits
		self.lc['upper_limit'] = np.zeros(len(self.lc))
		#Detections
		self.lc.mag[self.lc.Fratio >= self.lc.Fratio_err] = flux2mag(self.lc.Fratio[self.lc.Fratio >= self.lc.Fratio_err], self.lc.ZP[self.lc.Fratio >= self.lc.Fratio_err])
		self.lc.mag_err[self.lc.Fratio >= self.lc.Fratio_err] = dflux2dmag(self.lc.Fratio[self.lc.Fratio >= self.lc.Fratio_err], self.lc.Fratio_err[self.lc.Fratio >= self.lc.Fratio_err])
		self.lc.upper_limit[self.lc.Fratio >= self.lc.Fratio_err] = 99
		#Non-detections
		self.lc.mag[self.lc.Fratio < self.lc.Fratio_err] = 99
		self.lc.mag_err[self.lc.Fratio < self.lc.Fratio_err] = 99
		self.lc.upper_limit[self.lc.Fratio < self.lc.Fratio_err] = flux2mag(5*self.lc.Fratio_err[self.lc.Fratio < self.lc.Fratio_err], self.lc.ZP[self.lc.Fratio < self.lc.Fratio_err])
		return

	def match_ref_im(self):
		'''
		Match and add the reference image magnitude limit to each observation.
		'''
		self.text += '-----\nmatch reference images\n\n'
		self.lc['ref_maglim'] = ''
		for _ in self.lc.index:
			if 'g' in self.lc.obs_filter[_]:
				fid = 1
			elif 'r' in self.lc.obs_filter[_]:
				fid = 2
			elif 'i' in self.lc.obs_filter[_]:
				fid = 3
			else:
				self.text += 'ERROR: could not determine filter\n'
				continue
			try:
				self.lc.loc[_, 'ref_maglim'] = self.refmags[(
					(self.refmags.field==self.lc.fieldid[_]) &
					(self.refmags.fid==fid) & (self.refmags.rcid==self.lc.rcid[_]))
					].maglimit.iloc[0]
			except: #TEMPORARY SOLUTION!
				self.text += f'Could not find a ref mag for band {fid}, rcid {self.lc.rcid[_]}, field {self.lc.fieldid[_]}, using the average of the field & band\n'
				self.lc.loc[_, 'ref_maglim'] = self.refmags[(
					(self.refmags.field==self.lc.fieldid[_]) &
					(self.refmags.fid==fid))].maglimit.mean()
		return

	def select_baseline(self):
		'''
		Converts a text describing the baseline regions into a list of start & end mjd
		'''
		self.baselines = []
		#If there are multiple regions to be used they are separated by &
		for region in self.baseline_request.split (' & '):
			words = region.split(' ')
			if ((words[0].lower() == 'first') & (words[2].lower() == 'days')):
				base_mjd_start = self.lc.obsmjd.min()
				base_mjd_end = base_mjd_start + float(words[1])
			elif ((words[0].lower() == 'first') & (words[2].lower() == 'points')):
				base_mjd_start = self.lc.obsmjd.min()
				base_mjd_end = sort(self.lc.obsmjd.values)[int(words[1]) - 1]
			elif ((words[0].lower() == 'last') & (words[2].lower() == 'days')):
				base_mjd_end = self.lc.obsmjd.max()
				base_mjd_start = base_mjd_end - float(words[1])
			elif ((words[0].lower() == 'last') & (words[2].lower() == 'points')):
				base_mjd_start = sort(self.lc.obsmjd.values)[-int(words[1])]
				base_mjd_end = self.lc.obsmjd.max()
			elif words[0].lower() == 'from':
				base_mjd_start = float(words[1])
				if len(words) > 2:
					if words[2].lower() == 'to':
						base_mjd_end = float(words[3])
					else:
						base_mjd_end = self.lc.obsmjd.max()
				else:
					base_mjd_end = self.lc.obsmjd.max()
			elif words[0].lower() == 'to':
				base_mjd_start = self.lc.obsmjd.min()
				base_mjd_end = float(words[1])
			else:
				self.text += f'Baseline{region} not recognised, skipping\n'
				continue
			self.baselines.append([base_mjd_start, base_mjd_end])
		return


	def correct_baseline(self):
		self.text += '-----\ncorrect baseline \n\n'
		for band in ['g', 'r', 'i']:
			for field in self.lc[self.lc.obs_filter.str.contains(band)].fieldid.unique():
				for rcid in self.lc[((self.lc.obs_filter.str.contains(band)) &
									 (self.lc.fieldid == field))].rcid.unique():
					if self.mode == 'pre-ZTF': #Use requested region(s) for the baseline correction
						self.lc.reset_index(drop=True, inplace=True)
						wanted = [False]*len(self.lc)
						for r in self.baselines: #Loop over all baseline regions
							for _ in self.lc.index:
								if ((band in self.lc.loc[_, 'obs_filter']) &
									(self.lc.loc[_, 'fieldid'] == field) &
									(self.lc.loc[_, 'rcid'] == rcid) & (self.lc.loc[_, 'obsmjd'] >= r[0]) &
									(self.lc.loc[_, 'obsmjd'] <= r[1])):
									wanted[_] = True
						this_combo = self.lc[wanted]
						self.text += f'{len(this_combo)} points are in the baseline region for band {band}, field {field}, rcid{rcid}\n'
					else: #Assuming peak_mjd = correct, only use the points before base_gap days before the peak
						this_combo = self.lc[((self.lc.obs_filter.str.contains(band)) &
											  (self.lc.fieldid == field) &
											  (self.lc.rcid == rcid) &
											  (self.lc.obsmjd < self.peak_mjd-self.base_gap))]
					#Only use points that will be non-detections after the correction
					nondets = this_combo[(this_combo.Fratio-this_combo.Fratio.mean())/
										 this_combo.Fratio_err <= 5]
					if len(nondets) > 1: #Need at least 2 points for a baseline correction
						weights = 1/nondets.Fratio_err**2
						basel = sum(nondets.Fratio*weights)/sum(weights)
						#Weighted baseline uncertainties using the correct formula & the one with my typo from last iteration (for comparison)
						basel_err = np.sqrt(sum(weights* (nondets.Fratio - basel)**2) /
											(sum(weights)*(len(nondets)-1)))
						if self.mode == 'pre-ZTF':
							self.text += f'{band} {field} {rcid} {self.baselines}: basel = {basel:.3f}, err = {basel_err:.3f}, mean = {nondets.Fratio.mean():.3f}\n'
						else:
							self.text += f'{band} {field} {rcid} <{self.peak_mjd-self.base_gap}: basel = {basel:.3f}, err = {basel_err:.3f}, mean = {nondets.Fratio.mean():.3f}\n'
						#Correct Fratio
						self.lc.loc[self.lc[((self.lc.obs_filter.str.contains(band)) &
									(self.lc.fieldid == field) & (self.lc.rcid == rcid))].index,
									'Fratio'] -= basel
						self.lc.loc[self.lc[((self.lc.obs_filter.str.contains(band)) &
											 (self.lc.fieldid == field) & (self.lc.rcid == rcid))].index,
									'Fratio_err'] = np.sqrt(self.lc[((self.lc.obs_filter.str.contains(band))&
																	 (self.lc.fieldid == field)&
																	 (self.lc.rcid == rcid))].Fratio_err**2 +
																	 basel_err**2)
						#Correct mag
						self.lc.loc[self.lc[((self.lc.obs_filter.str.contains(band)) &
									(self.lc.fieldid == field) & (self.lc.rcid == rcid) &
									(self.lc.Fratio >= 5*self.lc.Fratio_err))].index,
									'mag'] = flux2mag(self.lc.loc[self.lc[((self.lc.obs_filter.str.contains(band)) &
																  (self.lc.fieldid == field) &
																  (self.lc.rcid == rcid) &
																  (self.lc.Fratio >= 5*self.lc.Fratio_err))].index,
																  'Fratio'], self.lc.loc[self.lc[((self.lc.obs_filter.str.contains(band)) &
																				(self.lc.fieldid == field) &
																				(self.lc.rcid == rcid) &
																				(self.lc.Fratio >= 5*self.lc.Fratio_err))].index,
																				'ZP'])
						self.lc.loc[self.lc[((self.lc.obs_filter.str.contains(band)) &
									(self.lc.fieldid == field) & (self.lc.rcid == rcid) &
									(self.lc.Fratio >= 5*self.lc.Fratio_err))].index,
									'mag_err'] = dflux2dmag(self.lc.loc[self.lc[((self.lc.obs_filter.str.contains(band)) &
																		(self.lc.fieldid == field) &
																		(self.lc.rcid == rcid) &
																		(self.lc.Fratio >= 5*self.lc.Fratio_err))].index,
																		'Fratio'],
															self.lc.loc[self.lc[((self.lc.obs_filter.str.contains(band)) &
																		(self.lc.fieldid == field) &
																		(self.lc.rcid == rcid) &
																		(self.lc.Fratio >= 5*self.lc.Fratio_err))].index,
																		'Fratio_err'])
						self.lc.loc[self.lc[((self.lc.obs_filter.str.contains(band)) &
									(self.lc.fieldid == field) & (self.lc.rcid == rcid) &
									(self.lc.Fratio < 5*self.lc.Fratio_err))].index,
									'upper_limit'] = flux2mag(5*self.lc.loc[self.lc[((self.lc.obs_filter.str.contains(band)) &
																			(self.lc.fieldid == field) &
																			(self.lc.rcid == rcid) &
																			(self.lc.Fratio < 5*self.lc.Fratio_err))].index,
																			'Fratio_err'], 30)
						self.lc.loc[self.lc[((self.lc.obs_filter.str.contains(band)) &
									(self.lc.fieldid == field) & (self.lc.rcid == rcid) &
									(self.lc.Fratio < 5*self.lc.Fratio_err))].index,
									'mag'] = 99
						self.lc.loc[self.lc[((self.lc.obs_filter.str.contains(band)) &
									(self.lc.fieldid == field) & (self.lc.rcid == rcid) &
									(self.lc.Fratio < 5*self.lc.Fratio_err))].index,
									'mag_err'] = 99
						self.lc.loc[self.lc[((self.lc.obs_filter.str.contains(band)) &
									(self.lc.fieldid == field) & (self.lc.rcid == rcid) &
									(self.lc.Fratio >= 5*self.lc.Fratio_err))].index,
									'upper_limit'] = 99
						self.text += f'The combination of filter {band}, field {field}, and rcid {rcid} has baseline correction {basel} +- {basel_err}\n'
						self.imp_df = self.imp_df.append({'name':f'b{band}f{field}rcid{rcid}', 'val':basel},
														 ignore_index=True)
						self.imp_df = self.imp_df.append({'name':f'b{band}f{field}rcid{rcid}_err', 'val':basel_err},
														 ignore_index=True)
					else:
						self.text += f'The combination of filter {band}, field {field}, and rcid {rcid} only has {len(nondets)} points available for the baseline correction. These observations are removed\n'
						self.lc.drop(self.lc[((self.lc.obs_filter.str.contains(band))&
											  (self.lc.fieldid == field)&
											  (self.lc.rcid == rcid))].index, inplace=True)
		return

	def calc_cloudy(self):
		#Cloudy has 4 conditions, pass 1 & the point should be removed
		#(Conditions from Adam Miller / Mat Smith)
		#They get values 1,2,4,8 to keep track of which condition removed each point,
		#& 16 for no_data (field=-99)
		# Define cloudy; default to the IPAC values
		airmass_slope = [0.20, 0.15, 0.07]
		zprms_cut     = [0.06, 0.05, 0.06]
		nmatch_cut    = [80, 120, 100]
		magzp         = self.lc.magzp - 2.5*np.log10(self.lc.exptime/30)
		magzp_cut     = [26.8, 26.75, 26.1]
		#Find the cloudy points
		cloudy = np.zeros(len(self.lc))
		#1) magzp > val I - val II * airmass      Wrong zp
		cloudy[np.where(((self.lc.obs_filter.str.contains('g')) & \
						 (magzp > magzp_cut[0]-airmass_slope[0]*self.lc.airmass)) | \
						((self.lc.obs_filter.str.contains('r')) & \
						 (magzp > magzp_cut[1]-airmass_slope[1]*self.lc.airmass)) | \
						((self.lc.obs_filter.str.contains('i')) & \
						 (magzp > magzp_cut[2]-airmass_slope[2]*self.lc.airmass)))] += 1
		#2) magzprms > val III                    Spread in zp too large
		cloudy[np.where(((self.lc.obs_filter.str.contains('g')) & (self.lc.magzprms > zprms_cut[0])) | \
						((self.lc.obs_filter.str.contains('r')) & (self.lc.magzprms > zprms_cut[1])) | \
						((self.lc.obs_filter.str.contains('i')) & (self.lc.magzprms > zprms_cut[2])))] += 2
		#3) nmatches < val IV                     Not enough catalog sources matched for alignment etc.
		cloudy[np.where(((self.lc.obs_filter.str.contains('g')) & (self.lc.nmatches < nmatch_cut[0])) | \
						((self.lc.obs_filter.str.contains('r')) & (self.lc.nmatches < nmatch_cut[1])) | \
						((self.lc.obs_filter.str.contains('i')) & (self.lc.nmatches < nmatch_cut[2])))] += 4
		#4) magzp < zp_rcid - val II * airmass    Wrong zp
		for _ in self.lc.rcid.unique():
			cloudy[np.where(((self.lc.obs_filter.str.contains('g')) & (self.lc.rcid == _) & \
							 (magzp < self.zp_rcid.g.iloc[int(_)]-airmass_slope[0]*self.lc.airmass)) | \
							((self.lc.obs_filter.str.contains('r')) & (self.lc.rcid == _) & \
							 (magzp < self.zp_rcid.r.iloc[int(_)]-airmass_slope[1]*self.lc.airmass)) | \
							((self.lc.obs_filter.str.contains('i')) & (self.lc.rcid == _) & \
							 (magzp < self.zp_rcid.i.iloc[int(_)]-airmass_slope[2]*self.lc.airmass)))] += 8
		#5) field = -99                           No data
		cloudy[np.where(self.lc.fieldid==-99)] += 16
		self.lc['cloudy'] = cloudy
		return

	def apply_cuts(self):
		'''
		Quality cuts applied:
		1.    flux_err is unphysically small (<2)
		2.    chi2 of fit is large (>3)
		4.    infofits failure
		8.    cloudy criteria failed
		'''
		#Flag all data points as good or bad
		self.lc['flag'] = 0
		ampl_err_min   = 2.001
		ampl_s2n_max   = 1e5
		ampl_err_max   = 1e6
		seeing_max     = 3     # Note IPAC recommends a cut at 4
		airmass_max    = 2
		moonillf_max   = 0.5
		field_max      = 999
		self.lc['ampl_s2n'] = self.lc.ampl/self.lc.ampl_err
		self.lc.loc[(self.lc.ampl_err<ampl_err_min) | (self.lc.ampl_s2n>ampl_s2n_max) | \
					(self.lc.ampl_err>ampl_err_max) | (np.isnan(self.lc.ampl_err)), 'flag'] += 1
		self.lc.loc[self.lc.chi2dof>3, 'flag'] += 2
		#Unset non-terminal failure bits
		unset_bits = [25, 22] # NB: unset_bits must be in reverse chronological order.
		for bit in unset_bits:
			self.lc.infobits[self.lc.infobits>=2**bit] -= 2**bit
		self.lc.loc[self.lc.infobits>0,'flag'] += 4
		self.lc.loc[self.lc.cloudy>0, 'flag'] += 8
		self.lc.loc[self.lc['seeing']>seeing_max, 'flag'] += 64
		#Apply the cuts
		return

	def renormalise_fpbot_lc(self, new_ZP=30.):
		# Take the input light-curve; determine updated fluxes
		# We do this for ampl and sigma
		self.lc.Fratio, self.lc.Fratio_err = self.renormalise_flux(self.lc.ampl.values, self.lc.ampl_err.values, self.lc.magzp.values)
		self.lc.sigma, self.lc.sigma_err = self.renormalise_flux(self.lc.sigma.values, self.lc.sigma_err.values, self.lc.magzp.values)
		self.lc['ZP'] = new_ZP
		# ----------- #
		# Calculate limiting magnitude: this uses the statistical only flux_errors.
		# Also Fix objects with no ZP_err.
		# ----
		# Note that our estimate differs from the header by 0.03-0.05
		# We add 1e-10 in quadrature to deal with things with flux_err==0.
		# ----------- #
		S2N_threshold = 5
		self.lc['mag_lim'] = self.lc.apply(lambda row: -2.5*np.log10(S2N_threshold*row['Fratio_err'])+row['ZP'] if \
							row['Fratio']/(np.sqrt(row['Fratio_err']**2+1e-10**2))<S2N_threshold else float("NAN"), axis=1)
		self.lc['magzpunc'] = self.lc.apply(lambda row: 1e-5 if row['magzpunc']<1e-5 else row['magzpunc'], axis=1)
		return

	def renormalise_flux(self, flux, flux_err, zp, new_ZP=30.):
		flux_new    = flux*10**(-0.4*(zp - new_ZP))
		fluxerr_new = flux_new*flux_err/flux            # sigma_f/f = sigma_f_new/f_new
		return flux_new, fluxerr_new

	def find_peak_date(self):
		'''
		Determine the date of peak light.
		Assume the highest Fratio observation is the peak, check for another
		observation with at least 1/2 times the peak flux within close_obs_size
		days. If none are found it is not considered real. Drop and try again.
		The chosen datapoints need to be detections, not upper limits
		'''
		self.text += '-----\nfind peak date\n\n'
		data = self.lc[((self.lc.flag&sum(self.cuts)==0) & (self.lc.mag!=99))].copy()
		if data.empty:
			self.text += 'Not enough points to get peak date & do binning!\n'
			self.peak_mjd = 0
			self.peak_mag = 0
			return
		if len(data) == 1: #If there is only 1 data point, use that one
			self.text += 'Only 1 observation found, it is automatically set at the peak mjd'
			self.peak_mjd = data.obsmjd.values[0]
			self.peak_mag = data.mag.values[0]
			return
		close_obs_size = 10 #Nr. of days considered to be close by
		peak_light = data.obsmjd[data.Fratio.idxmax()]
		close_obs = data[abs(data.obsmjd-peak_light)<close_obs_size]
		while len(close_obs.Fratio[close_obs.Fratio>
				  0.5*close_obs.Fratio.max()])<2:
			data = data.drop([data.Fratio.idxmax()], axis=0)
			peak_light = data.obsmjd[data.Fratio.idxmax()]
			close_obs = data[abs(data.obsmjd-peak_light)<close_obs_size]

			#Not enough points to determine peak realness? Use last guess
			if len(data)<2:
				self.text += 'Could not verify peak_light, using last guess\n'
				break
		self.peak_mjd = peak_light
		self.peak_mag = data.mag[data.Fratio.idxmax()]
		self.text += f'found peak at mjd {self.peak_mjd} at mag {self.peak_mag}\n'
		return

	def check_refmjd(self):
		#Make a note of band, field, rcid combinations where the SN might be in the ref image
		for band in [['g', 1], ['r', 2], ['i', 3]]:
			for field in self.lc.fieldid.unique():
				for rcid in self.lc.rcid.unique():
					ref = self.refmags[((self.refmags.fid==band[1])&
										(self.refmags.field==field)&
										(self.refmags.rcid==rcid))]
					if self.mode == 'pre-ZTF':
						#Note the periods in which the references were made & if that is (partially) in or outside the used baseline region
						for i in range(len(ref)):
							self.text += f'For band {band[0]}, field {field}, rcid {rcid}, reference {i+1} was made between {ref.startmjd.values[i]:.2f} and {ref.endmjd.values[i]:.2f}\n'
							in_baseline = 'outside'
							for _ in self.baselines:
								if ((ref.startmjd.values[i] >= _[0]) & (ref.startmjd.values[i] <= _[1])):
									if ((ref.endmjd.values[i] >= _[0]) & (ref.endmjd.values[i] <= _[1])):
										in_baseline = 'completely in'
										break
									else:
										in_baseline = 'partially in '
										break
								elif ((ref.endmjd.values[i] >= _[0]) & (ref.endmjd.values[i] <= _[1])):
									in_baseline = 'partially'
									break
							self.text += f'This is {in_baseline} the chosen baseline region\n'
					else:
						if len(ref)==1:
							if ((ref.startmjd.values[0]<self.peak_mjd)&(ref.endmjd.values[0]>self.peak_mjd)):
								self.text +=f'WARNING: band {band[0]}, field {field}, rcid {rcid} might have the SN in the ref image\n'
						elif len(ref)>1:
							self.text +=f'WARNING: multiple instances of band {band[0]}, field {field}, rcid {rcid}, cannot determine if the SN might be inthe ref image\n'
		return

	def bin_all(self):
		#Bin the late-time data & store it in bins for all band, binsize & filter combinations
		for band in ['g', 'r', 'i']:
			for binsize in self.binsizes:
				for phase in self.phases:
					#Pick only 1 band at a time & only use the late-time data for binning
					self.binlist.append(bin_late_time(self.lc[((self.lc.obs_filter.str.contains(band)) &
															   (self.lc.obsmjd>=
																self.peak_mjd+self.late_time))],
													  'ZTF_'+band, binsize, phase, self.method,
													  self.late_time+self.peak_mjd))
		return

	def check_bins(self):
		#Check all bins and give a verdict to each binning attempt
		self.text += '-----\nGive verdicts\n\n'
		counter = 0
		for attempt in self.binlist:
			#Make sure there are bins to check
			if len(attempt)==0:
				self.text += f'There are no bins for binning attempt {counter}\n'
				counter += 1
				continue
			counter += 1
			#Select the points that can be used for the tail fit if needed
			#These should lie between the earliest allowed start & late-time observations
			if self.mode == 'pre-ZTF':
				points = None
			else:
				points = self.lc[((self.lc.obs_filter.str.contains(attempt.obs_filter[0][-1]))&
								  (self.lc.obsmjd>self.peak_mjd+self.earliest_tail_fit_start)&
								  (self.lc.obsmjd<self.peak_mjd+self.late_time))]
			#Give the verdict and append it to the list
			verdict, err = give_verdict(attempt, points, self.peak_mjd, self.verdict_sigma,
										self.tail_fit_chi2dof_tresh, self.saveloc, self.save_lc_and_bins,
										self.final_mjd[attempt.obs_filter[0][-1]])
			self.verdictlist.append(verdict)
			#Log the fitting error if it occured
			if err is not None:
				self.text += err +'\n'
		return


#*----------------*
#| Main functions |
#*----------------*

def check_object(args):
	#Load the light curve
	obj_data = ztf_object(args)
	#Check if there is data to do the binning, end if not
	if obj_data.lc.empty:
		obj_data.text += 'It seems that the dataframe has come up empty, no binning could be performed\n'
	elif len(obj_data.lc[obj_data.lc.obsmjd>=obj_data.peak_mjd+obj_data.late_time]) > 0:
		#Do the binning for each combination of filter, binsize & phase
		obj_data.bin_all()
		#Save the bins
		all_bins = pd.concat(obj_data.binlist, ignore_index=True)
		if obj_data.save_lc_and_bins:
			all_bins.to_csv(obj_data.saveloc / 'bins.csv', index=False)
		#Run the filtering program
		obj_data.check_bins()
		#Save the result of the filtering program
		all_verdicts = pd.concat(obj_data.verdictlist, ignore_index=True)
		all_verdicts.to_csv(obj_data.saveloc / 'verdicts.csv', index=False)
	else:
		obj_data.text += 'Unfortunately, there are no useful late-time observations available. Either they do not exist, or they have been cut for some reason\nBinning could not be performed\n'
	#Save the notes & imp_df (Disabled in real_time mode)
	if obj_data.mode != 'real_time':
		obj_data.imp_df.to_csv(obj_data.saveloc / 'derived_values.csv', index=False)
		ftext = open(obj_data.saveloc/'notes.txt', 'w')
		ftext.write(obj_data.text)
		ftext.close()
	return

def give_final_verdicts(mode, obj_list, host_dat, min_sep, min_successes, note_missing_host=False):
	#Give a final verdict for each object & save them
	final_verdicts = pd.DataFrame()
	for _ in obj_list:
		if None in host_dat: #For the simulations & pre-ZTF
			final_verdicts = final_verdicts.append(final_verdict(_, [None],
																 min_sep, min_successes, mode),
												   ignore_index=True)
		else:
			try:
				final_verdicts = final_verdicts.append(final_verdict(_, host_dat[host_dat.ztfname==_.name],
																	 min_sep, min_successes, mode,
																	 note_missing_host=note_missing_host),
													   ignore_index=True)
			except:
				print('Error: host data not recognised, assuming none to create final verdicts')
				final_verdicts = final_verdicts.append(final_verdict(_, [None],
																	 min_sep, min_successes, mode),
													   ignore_index=True)
	return final_verdicts


#*-------------------*
#| Binning functions |
#*-------------------*

def bin_late_time(data, band, binsize, phase, method, late_time):
	'''
	Bin the data &
	
	Parameters:
	data (DataFrame): data to bin
	binsize (float): Size of used bins
	phase (float): location of first object relative to the 1st bin
	method (int): How to place the bins (1st datapoint always in 1st bin)
		1: blindly place bins one after another

		2: start each new bin at the next datapoint
		If the space between the end of last bin & next datapoint > binsize:
		1st datapoint is at the start_phase in its bin
		(Avoid nullifying this setting after a large gap in observations)

		3: If there are less than 3 objects in the next bin, and they
		are within the 1st 10% of that bin, increase the binsize of this bin
		to include them (prevents bins with unnaturally low sigma)

		4: Combine methods 2 & 3
	Returns:
	results (DataFrame): The filled bins
	'''
	#Initialize DataFrame, bin counter, & 1st bin left side
	result = pd.DataFrame(columns=['obs_filter', 'binsize', 'phase', 'method',
								   'mjd_start', 'mjd_stop', 'mjd_bin', 'Fratio',
								   'Fratio_std', 'nr_binned', 'significance',
								   'Fratio_std_no_syst_err'])
	counter = 0
	newbin_start =  data.obsmjd.min() - binsize*phase
	#Fill the bins 1 by 1
	while newbin_start < data.obsmjd.max():
		#Calc right side of this bin according to the chosen method
		newbin_stop = newbin_start + binsize
		if (((method == 3) | (method == 4)) &
				(len(data[(data.obsmjd>=newbin_stop) &
						  (data.obsmjd<newbin_stop+0.1*binsize)])!=0) &
				(len(data[(data.obsmjd>=newbin_stop) &
						  (data.obsmjd<newbin_stop+0.1*binsize)])<3) &
				(len(data[(data.obsmjd>=newbin_stop+0.1*binsize) &
						  (data.obsmjd<newbin_stop+2*binsize)])==0)):
			newbin_stop = data.obsmjd[(data.obsmjd>=newbin_stop) &
									  (data.obsmjd<newbin_stop+0.1*binsize)].max() + 1e-7
		#Make sure the new bin starts at late_time or later
		if newbin_start < late_time:
			newbin_start = late_time
		#Select data
		thisbin = data[((data.obsmjd>=newbin_start) & (data.obsmjd<newbin_stop))]
		#If the bin isn't empty, fill it
		if len(thisbin)!=0:
			#Calc bin params
			weights = 1./thisbin.Fratio_err**2
			fratio = sum(thisbin.Fratio*weights)/sum(weights)
			mjd_bin = sum(thisbin.obsmjd*weights)/sum(weights)
			if len(thisbin)==1:
				std_dev = thisbin.Fratio_err.max()
			else:
				std_dev = np.sqrt(sum(weights * (thisbin.Fratio-fratio)**2)
								  / (sum(weights) * (len(thisbin)-1)))
			#Add systematic uncertainties coming from the reference images
			#NOTE std_dev = without syst.error, std_dev_full = with syst.error
			#ref_maglim = 5 sigma mag limit --> sigma_ref = 1/5 * ref_fratiolim
			std_dev_full = np.sqrt(std_dev**2 + np.median(10**(-0.4*(thisbin.ref_maglim-30))/5)**2)
			if std_dev == 0:		#If this happens, don't trust bin
				signif= 0
			else:
				signif = fratio/std_dev_full
			#Store in the result & update the counter
			result.loc[counter] = [band, binsize, phase, method, newbin_start,
								   newbin_stop, mjd_bin, fratio, std_dev_full,
								   len(thisbin), signif, std_dev]
			counter += 1
		#Calc left side of next bin according to the chosen method
		if ((method == 1) | (method == 3)):
			newbin_start = newbin_stop
		else: #Method = 2 or 4
			#If the last datapoint is binned, break out of the loop
			if len(data[data.obsmjd>=newbin_stop]) == 0:
				break
			next_obs = data.obsmjd[
				data.obsmjd>=newbin_stop].min()
			#If an empty bin can fit between the end of the current & start of the next bin
			#reapply the phase offset
			if next_obs < newbin_stop + binsize:
				newbin_start = next_obs
			else:
				newbin_start = next_obs - binsize*phase
	return result

#*---------------------*
#| Filtering functions |
#*---------------------*

def give_verdict(bins, points, peak_mjd, sigma, chi2dof_tresh, saveloc, save_lc_and_bins, final_obs_mjd):
	err = None
	fit = None
	max_val = None
	verdict = 0
	last_bin_det = False
	#Are there any bins?
	if len(bins) > 0:
		#Are there any bins that are detections?
		if bins.significance.max() < sigma:
			verdict += 1
		else:
			verdict += 2
			#Trusted detections require Fratio >0, significance >= sigma
			trusted_bins = bins[((bins.Fratio>0) & (bins.significance>=sigma))]
			#Do the bins with detections contain enough points? ##I think I removed this condition. If so the comment should be removed as well
			if not trusted_bins.empty:
				verdict += 4
				#Is the last bin a detection?
				if ((bins.Fratio.iloc[-1] > 0) & (bins.significance.iloc[-1] >= sigma)):
					last_bin_det = True
				#Is it a single bin or are there multiple ones?
				if len(trusted_bins) == 1:
					verdict += 8
				#If there are multiple bins, are they ajacent to one another?
				elif 1 in (trusted_bins.index[1:]-trusted_bins.index[:-1]):
					verdict += 16
					#Is the first bin a detection? (Only interested if there are ajacent dets)
					if trusted_bins.index[0] == bins.index[0]:
						verdict += 32
						if points is not None:
							#Try to fit a nuclear decay tail model to the data
							try:
								fit, max_val = test_tail_model(bins, points, peak_mjd)
								#Was the fit successful enough to estimate errorbars?
								if not fit.errorbars:
									verdict += 128
								else:
									verdict += 256
									#Is the final fit good enough (below the reduced chi square treshold)?
									if fit.redchi > chi2dof_tresh:
										verdict += 512
									else:
										verdict += 1024
									#Is the half-life within the expected range, not larger?
									if fit.params['tau'].value-sigma*fit.params['tau'].stderr > fit.init_values['tau']:
										verdict += 2048
									else:
										verdict += 4096
							except: #If the fit ends up with an error, make note of it and skip the fit evaluation part
								err = traceback.format_exc()
								verdict += 64
	# Check if the last bin was a detection
	final_bin_det = True if bins.significance.values[-1] >= sigma else False
	#Put the results in a df, including the fit result if it was made
	if fit != None:
		#Save the model
		x = np.linspace(fit.params['t0'].value, bins.mjd_bin.max()+100, 3000)
		fit_res = pd.DataFrame({'x':x, 'y':fit.eval(x=x)*max_val, 'dy':fit.eval_uncertainty(x=x)*max_val})
		if save_lc_and_bins:
			fit_res.to_csv(saveloc/f'tail_mod_f{bins.obs_filter[0][-1]}_b{bins.binsize[0]}_p{bins.phase[0]}.csv')
		if fit.errorbars:
			result = pd.DataFrame({'band':[bins.obs_filter[0]], 'binsize':[bins.binsize[0]],
								   'phase':[bins.phase[0]], 'a_init':[fit.params['a'].init_value*max_val],
								   'a':[fit.params['a'].value*max_val],
								   'sigma_a':[fit.params['a'].stderr*max_val],
								   't_half_init':[fit.params['tau'].init_value],
								   't_half':[fit.params['tau'].value],
								   'sigma_t_half':[fit.params['tau'].stderr],
								   't0':[fit.params['t0'].value], 'chi2':[fit.chisqr],
								   'chi2dof':[fit.redchi], 'AIC':[fit.aic], 'BIC':[fit.bic],
								   'verdict':[verdict], 'final_bin_det':[final_bin_det],
								   'final_obs_mjd':[final_obs_mjd]})
		else:
			result = pd.DataFrame({'band':[bins.obs_filter[0]], 'binsize':[bins.binsize[0]],
								   'phase':[bins.phase[0]], 'a_init':[fit.params['a'].init_value*max_val],
								   'a':[fit.params['a'].value*max_val], 'sigma_a':[' '],
								   't_half_init':[fit.params['tau'].init_value],
								   't_half':[fit.params['tau'].value], 'sigma_t_half':[' '],
								   't0':[fit.params['t0'].value], 'chi2':[fit.chisqr],
								   'chi2dof':[fit.redchi], 'AIC':[fit.aic], 'BIC':[fit.bic],
								   'verdict':[verdict], 'final_bin_det':[final_bin_det],
								   'final_obs_mjd':[final_obs_mjd]})
	else:
		result = pd.DataFrame({'band':[bins.obs_filter[0]], 'binsize':[bins.binsize[0]],
							   'phase':[bins.phase[0]], 'a_init':[' '], 'a':[' '], 'sigma_a':[' '],
							   't_half_init':[' '], 't_half':[' '], 'sigma_t_half':[' '],
							   't0':[' '], 'chi2':[' '], 'chi2dof':[' '], 'AIC':[' '],
							   'BIC':[' '], 'verdict':[verdict], 'final_bin_det':[final_bin_det],
							   'final_obs_mjd':[final_obs_mjd]})
	return result, err

def final_verdict(obj_loc, host_dat, min_sep, min_successes, mode, note_missing_host=False):
	#Give the final verdict of the object at the given location
	dets = ' '
	lone = ' '
	adjacent = ' '
	pos_tail = ' '
	fit_failed = ' '
	normal = ' '
	too_nuc = ' '
	suc_g = False
	suc_r = False
	suc_i = False
	nr_g = 0
	nr_r = 0
	nr_i = 0
	final_bin_det_g = 0
	final_bin_det_r = 0
	final_bin_det_i = 0
	last_obs_mjd_g = 0
	last_obs_mjd_r = 0
	last_obs_mjd_i = 0
	if not (obj_loc / 'verdicts.csv').is_file():
		#No verdicts given (Probably no bins made)
		bins = False
	else:
		bins = True
		attempts = pd.read_csv(obj_loc / 'verdicts.csv')
		if max(attempts.verdict.values&4)==0: #Are there detections?
			dets = False
		else:
			dets=True
			bel_verdicts = attempts[attempts.verdict&4!=0] #Only keep the believable attempts
			if True not in [i&8==0 for i in bel_verdicts.verdict.values]: #Are there multiple binned detections?
				lone = True
			else:
				lone = False
				if True not in [i&16!=0 for i in bel_verdicts.verdict.values]: #Are some detections adjacent?
					adjacent = False
				else:
					adjacent = True
					if True in [i&32!=0 for i in bel_verdicts.verdict.values]: #Is the 1st bin a detection? (possible tail)
						pos_tail = True
						with_fits = bel_verdicts[((bel_verdicts.verdict&32!=0) & (bel_verdicts.verdict&16!=0))] #Only the objects with tail fits (No tail fit but still pass has to have verdict==22)
						if True not in [i&64==0 for i in with_fits.verdict]: #Did the fit always fail?
							fit_failed = True
							#Failed fits are interesting --> successfull attempts have verdict are 22 or contain 64
							if ((host_dat.separation.values[0]<min_sep) & (host_dat.separation.values[0]>-99)): #Is it too close to the host nucleus?
								too_nuc = True
							else:
								too_nuc = False
								succes_attempts = bel_verdicts[((bel_verdicts.verdict&64!=0) | (bel_verdicts.verdict==22))]
						else:
							fit_failed = False
							if False not in [((i&1024!=0) & (i&4096!=0)) for i in with_fits.verdict.values]: #Are any of the fits inconsistent with a normal Ia tail?
								normal = True
								#Only failed fits might be interesting --> successfull attempts have verdict are 22 or contain 64
								if None in host_dat: #When running the simulations
									succes_attempts = bel_verdicts[((bel_verdicts.verdict&64!=0) | (bel_verdicts.verdict==22))]
								elif len(host_dat) == 0: #If there is no host_dat, assume its far away from the host
									too_nuc = False
									if note_missing_host:
										print(obj_loc.name, 'has no host data')
									succes_attempts = bel_verdicts[((bel_verdicts.verdict&64!=0) | (bel_verdicts.verdict==22))]
								else:
									if ((host_dat.separation.values[0]<min_sep) & (host_dat.separation.values[0]>-99)): #Is it too close to the host nucleus?
										too_nuc = True
									else:
										too_nuc = False
										succes_attempts = bel_verdicts[((bel_verdicts.verdict&64!=0) | (bel_verdicts.verdict==22))]
							else:
								normal = False
								#Failed & non normal Ia fits are interesting --> successfull attempts are those with verdict 22 or with_fits not containing 1024 or 4096
								if None in host_dat: #When running the simulations
									too_nuc = False
									succes_attempts = pd.concat([bel_verdicts[bel_verdicts.verdict==22],
																 with_fits[((with_fits.verdict&1024==0)|
																			(with_fits.verdict&4096==0))]],
																 ignore_index=True)
								elif len(host_dat) == 0: #If there is no host_dat, assume its far away from the host
									too_nuc = False
									if note_missing_host:
										print(obj_loc.name, 'has no host data')
									succes_attempts = pd.concat([bel_verdicts[bel_verdicts.verdict==22],
																 with_fits[((with_fits.verdict&1024==0)|
																			(with_fits.verdict&4096==0))]],
																 ignore_index=True)
								else:
									if ((host_dat.separation.values[0]<min_sep) & (host_dat.separation.values[0]>-99)): #Is it too close to the host nucleus?
										too_nuc = True
									else:
										too_nuc = False
										succes_attempts = pd.concat([bel_verdicts[bel_verdicts.verdict==22],
																	 with_fits[((with_fits.verdict&1024==0)|
																	 			(with_fits.verdict&4096==0))]],
																	 ignore_index=True)
					else:
						pos_tail = False
						if None in host_dat: #When running the simulations
							too_nuc = False
							succes_attempts = bel_verdicts[bel_verdicts.verdict==22]

						elif len(host_dat) == 0: #If there is no host_dat, assume its far away from the host
							too_nuc = False
							if note_missing_host:
								print(obj_loc.name, 'has no host data')
							succes_attempts = bel_verdicts[bel_verdicts.verdict==22]
						else:
							if ((host_dat.separation.values[0]<min_sep) & (host_dat.separation.values[0]>-99)): #Is it too close to the host nucleus?
								too_nuc = True
							else:
								too_nuc = False
								succes_attempts = bel_verdicts[bel_verdicts.verdict==22] #Only successful attempts through this route
		if not too_nuc:
			#Are there enough successful attempts per band?
			nr_g = len(succes_attempts[succes_attempts.band=='ZTF_g'])
			nr_r = len(succes_attempts[succes_attempts.band=='ZTF_r'])
			nr_i = len(succes_attempts[succes_attempts.band=='ZTF_i'])
			if nr_g >= min_successes:
				suc_g = True
			if nr_r >= min_successes:
				suc_r = True
			if nr_i >= min_successes:
				suc_i = True
		if len(attempts[attempts.band=='ZTF_g']) > 0:
			final_bin_det_g = len(attempts[((attempts.band=='ZTF_g') & (attempts.final_bin_det==True))])
			last_obs_mjd_g = attempts[attempts.band=='ZTF_g'].final_obs_mjd.values[0]
		if len(attempts[attempts.band=='ZTF_r']) > 0:
			final_bin_det_r = len(attempts[((attempts.band=='ZTF_r') & (attempts.final_bin_det==True))])
			last_obs_mjd_r = attempts[attempts.band=='ZTF_r'].final_obs_mjd.values[0]
		if len(attempts[attempts.band=='ZTF_i']) > 0:
			final_bin_det_i = len(attempts[((attempts.band=='ZTF_i') & (attempts.final_bin_det==True))])
			last_obs_mjd_i = attempts[attempts.band=='ZTF_i'].final_obs_mjd.values[0]
	if mode != 'pre-ZTF':
		df = pd.DataFrame({'name':obj_loc.name, 'bins':bins, 'detections':dets, 'lone_bins':lone,
						   'adjacent':adjacent, 'possible_tail':pos_tail, 'fit_always_fails':fit_failed,
						   'normal_tail':normal, 'too_nuclear':too_nuc, 'suc_g':suc_g, 'suc_r':suc_r,
						   'suc_i':suc_i, 'nr_suc_g':nr_g, 'nr_suc_r':nr_r, 'nr_suc_i':nr_i,
						   'final_bin_det_g':final_bin_det_g, 'last_obs_mjd_g':last_obs_mjd_g,
						   'final_bin_det_r':final_bin_det_r, 'last_obs_mjd_r':last_obs_mjd_r,
						   'final_bin_det_i':final_bin_det_i, 'last_obs_mjd_i':last_obs_mjd_i}, index=[0])
	else:
		df = pd.DataFrame({'name':obj_loc.parent.name, 'baseline_region':obj_loc.name, 'bins':bins,
						   'detections':dets, 'lone_bins':lone, 'adjacent':adjacent,
						   'too_nuclear':too_nuc, 'suc_g':suc_g, 'suc_r':suc_r, 'suc_i':suc_i,
						   'nr_suc_g':nr_g, 'nr_suc_r':nr_r, 'nr_suc_i':nr_i,
						   'final_bin_det_g':final_bin_det_g, 'last_obs_mjd_g':last_obs_mjd_g,
						   'final_bin_det_r':final_bin_det_r, 'last_obs_mjd_r':last_obs_mjd_r,
						   'final_bin_det_i':final_bin_det_i, 'last_obs_mjd_i':last_obs_mjd_i}, index=[0])
	return df

#*------------------------*
#| Tail fitting functions |
#*------------------------*

def tail_model(x, a, tau, t0):
	#a = amplitude, tau = half-life, t0 = ref time
	return a * 2**(-(x-t0)/tau)

def test_tail_model(bins, points, peak_date):
	#Select the correct data to be used for the fit
	#use max 0.5x the amount of individual points compared to bins
	points_amount = int(len(bins)/2)
	Fratios = np.append(points.Fratio.values[-points_amount:], bins.Fratio.values)
	Fratio_errs = np.append(points.Fratio_err.values[-points_amount:], bins.Fratio_std.values)
	#Rescaling (makes fitting work better)
	max_val = Fratios.max()
	Fratios = Fratios/max_val
	Fratio_errs = Fratio_errs/max_val
	mjds = np.append(points.obsmjd.values[-points_amount:], bins.mjd_bin.values)
	#Check that there is indeed some data to fit, else quit
	if len(Fratios) == 0:
		print('No data to fit')
		return None, 0
	#Get an estimate for the initial guesses
	a = Fratios[0] #Value of the 1st datapoint
	tau = 50.      #half-life in a normal SN Ia
	t0 = mjds[0]   #date of the 1st datapoint
	#Make and fit the model, limit the parameters to allow only sensible values
	mod = Model(tail_model)
	params = mod.make_params()
	params['a'].set(value=a, min=0)
	params['tau'].set(value=tau, min=0)
	params['t0'].set(value=t0, vary=False)
	fit = mod.fit(Fratios, params, x=mjds, weights=1/Fratio_errs)
	#return the fit
	return fit, max_val

#*-----------------*
#| Other functions |
#*-----------------*

def get_host_data(objs, loc_list, dat_list):
	#Get host information for the objects
	sn_hosts = pd.read_csv(loc_list, sep=' ', header=0)
	#Select only the ones that are needed
	sn_hosts = sn_hosts[sn_hosts.ztfname.isin(objs)].copy()
	#Get the separation
	sn_hosts['separation'] = SkyCoord(sn_hosts.sn_ra, sn_hosts.sn_dec, unit=u.deg,
	                                  frame='icrs').separation(SkyCoord(sn_hosts.host_ra,
	                                                                    sn_hosts.host_dec,
	                                                                    unit=u.deg,
	                                                                    frame='icrs')).arcsecond
	#Those without a host location get separation=-99
	sn_hosts.loc[((sn_hosts.host_ra==270)&(sn_hosts.host_dec==-80)), 'separation'] = -99
	#add the catalog magnitudes for the host
	host_data = pd.read_csv(dat_list, header=0, usecols=['ztfname', 'ra_gal',
														 'dec_gal', 'host_cat_mag',
														 'host_cat_mag_err'])
	for i in sn_hosts[sn_hosts.host_ra!=270].ztfname.values:
	    host_mags = eval(host_data[host_data.ztfname==i].host_cat_mag.values[0])
	    for key in host_mags:
	        sn_hosts.loc[sn_hosts[sn_hosts.ztfname==i].index, key] = host_mags[key]
	return sn_hosts

def flux2mag(flux, zp):
	return -2.5*np.log10(flux) + zp

def dflux2dmag(flux, flux_err):
	return 2.5*flux_err / (np.log(10)*flux)


#*----------------------------------------------*
#| Functions specific for using the simulations |
#*----------------------------------------------*

def make_fpbot_df(lc, fields):
	#Format the lightcurve in an FPbot-style csv file
	df = pd.DataFrame()
	#Calculate derived values
	observatory = EarthLocation(lat=33.35627096604836*u.deg, lon=-116.86481294596469*u.deg, height=1700*u.m)
	mags, mag_errs, uplims = flux2mag_fpbot(np.array(lc['flux']), np.array(lc['fluxerr']),
									  np.array(lc['zp']))
	t = Time(lc['time'], format='mjd')
	airmass = calc_airmass([fields[fields.ID==i].RA.values[0] for i in lc['field']],
						   [fields[fields.ID==i].Dec.values[0] for i in lc['field']],
						   t, observatory)
	decday = [f'{i.isot[0:4]}{i.isot[5:7]}{i.isot[8:10]}{str(float(i.isot[11:13])/24+float(i.isot[14:16])/(24*60)+(float(i.isot[17:]))/(24*3600))[2:8]}' for i in t]
	#Put all values in the DataFrame
	df['obsmjd'] = lc['time']
	df['obs_filter'] = lc['band']
	df['ampl'] = lc['flux']
	df['Fratio'] = lc['flux']
	df['Fratio_err'] = lc['fluxerr']
	df['mag'] = mags
	df['mag_err'] = mag_errs
	df['upper_limit'] = uplims
	df['ampl_err'] = lc['fluxerr']
	df['chi2dof'] = 1
	df['seeing'] = 2
	df['magzp'] = [25.8 if i[-1]=='g' else 25.9 if i[-1]=='r' else 25.5 for i in lc['band']]
	df['magzpunc'] = 0.1
	df['magzprms'] = 0.04
	df['airmass'] = airmass
	df['nmatches'] = [130 if i>=19 else 20 for i in -2.5*np.log10(5*lc['fluxerr'])+lc['zp']]
	df['rcid'] = lc['ccd']
	df['fieldid'] = lc['field']
	df['infobits'] = 0
	df['exptime'] = 30
	df['target_x'] = 5
	df['target_y'] = 5
	df['ZP'] = lc['zp']
	df['filename'] = [f"ztf_{decday[i]}_{'%06.f'%lc['field'][i]}_z{lc['band'][i][-1]}_c{'%02.f'%int(lc['ccd'][i]/4+1)}_o.fits" for i in range(len(decday))]
	return df

def flux2mag_fpbot(flux, fluxerr, zp):
	#Convert flux, fluxerr to mag, magerr, upper limmit
	#Give 5 sigma upper limit when flux < 5*fluxerr, value = 99: not applicable
	mag = np.ones_like(flux)*99
	magerr = np.ones_like(flux)*99
	uplim = np.ones_like(flux)*99
	mask = flux>=5*fluxerr
	mag[mask] = -2.5*np.log10(flux[mask])+zp[mask]
	magerr[mask] = np.abs(-2.5*fluxerr[mask] / (flux[mask]*np.log(10)))
	uplim[~mask] = -2.5*np.log10(5*fluxerr[~mask])+zp[~mask]
	return mag, magerr, uplim

def calc_airmass(ra, dec, t, observatory):
	#Calculate the rmass at a given location & time
	pointings = SkyCoord(ra, dec, unit='deg')
	return [pointings[i].transform_to(AltAz(obstime=t[i], location=observatory)).secz for i in range(len(t))]

if (__name__ == "__main__"):
	main()
