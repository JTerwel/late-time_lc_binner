'''
A program to stack and display late-time observations from the ZTF SN Ia dataset

Author: Jacco Terwel
Date: 25-07-22

- Major overhaul of the program
- Rewritten binning part, removed obsolete functions, added filtering part
- Redesigned how results are saved, should be easier & clearer now
- Final version of the code
'''

#Imports & global constants
import numpy as np
from numpy import nan, inf
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm
from lmfit import Model, Parameters
import traceback

def main():
	'''
	Main function of the program.

	Make a list of locations to find the required data for each object, set
	the location where all results will be saved (separate folders for each
	object), and control the progress bar.
	'''
	#Set location where the results will be saved
	print('\nStarting the program\nCollecting objects')
	saveloc = Path("/Users/terwelj/Projects/Late-time_signals/Bin_results")

	#Set the location of the object csv files and list them
	#Only get the objects that are in my list as well as have lcs
	#--> 952 objects (includes non-2018 objects, and those that will fail)
	datafiles_all = list(Path("/Users/terwelj/Projects/Late-time_signals/ZTF18_Ia_sample_full+40_extra").rglob('*.csv'))
	obj_list = pd.read_csv('/Users/terwelj/Projects/Late-time_signals/ZTF18_Ia_names_10-02-2022.csv',
		header=None)
	datafiles = [i for i in datafiles_all if i.name.rsplit('_S',1)[0] in obj_list.values]

	#Load the reference image data
	refmags = pd.read_csv('/Users/terwelj/projects/Late-time_signals/ref_im_data.csv',
		usecols=['field', 'fid', 'rcid', 'maglimit', 'startmjd', 'endmjd'])


	#Load camera ZPs
	read_opts = {'delim_whitespace': True, 'names': ['g', 'r', 'i'],'comment': '#'}
	zp_rcid = pd.read_csv('/Users/terwelj/Projects/Late-time_signals/zp_thresholds_quadID.txt', **read_opts)

	#Set free parameters & make the arg_list
	late_time = 100
	cuts = {'ampl_err': 2, 'chi2dof': 3, 'seeing': 3, 'cloudy': 0, 'infobits': 0}
	base_gap = 40
	binsizes = [100,  75, 50, 25]
	phases = [0.0, 0.25, 0.5, 0.75]
	method = 4
	earliest_tail_fit_start = 60
	verdict_sigma = 5
	tail_fit_chi2dof_tresh = 5
	min_sep = 1 #Minimal SN - host nucleus separation in arcsec
	min_successes = 4
	cols = ['obsmjd', 'filter', 'Fratio', 'Fratio.err', 'mag', 'mag_err', 'upper_limit',
			'ampl.err', 'chi2dof', 'seeing', 'magzp', 'magzprms', 'airmass', 'nmatches',
			'rcid', 'fieldid', 'infobits', 'filename']
	args = [[f, refmags, saveloc, late_time, zp_rcid, cuts, base_gap, cols, binsizes, phases,
			 method, earliest_tail_fit_start, verdict_sigma, tail_fit_chi2dof_tresh] for f in datafiles]#I never actually use host_dat in this part --> remove it

	#Save all settings
	data_settings = cuts
	data_settings.update({'late_time':late_time, 'binsizes':binsizes, 'phases':phases,
						  'method':method, 'min_sep':min_sep, 'min_successes':min_successes})
	settings = pd.Series(data_settings)
	settings.to_csv(saveloc / 'settings.csv')

	print('Initialization complete, strarting checking each object')
	#use each cpu core separately & keep track of progress
	pool = mp.Pool(mp.cpu_count())

	list(tqdm(pool.imap_unordered(check_object, args), total=len(datafiles)))
	#list(tqdm(pool.imap_unordered(check_object, args[48:64]), total=16)) #For testing
	pool.close()
	pool.join()
	print('All objects binned & evaluated, putting objects through the final filter')
	#Load the host data
	loc_list = '/Users/terwelj/Projects/Late-time_signals/SN_host_locs.csv'
	dat_list = '/Users/terwelj/Projects/Late-time_signals/SN_host_data.csv'
	host_dat = get_host_data([i.name.rsplit('_S',1)[0] for i in datafiles], loc_list, dat_list)
	#Give the final verdicts
	all_obs = [f for f in saveloc.rglob('*') if f.is_dir()]
	final_verdicts = give_final_verdicts(all_obs, host_dat, min_sep, min_successes)
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
	- cols (list): list of lc columns to read in
	- binsizes (list): binsizes to use when binning
	- phases (list): offsets to the 1st bin to use when binning
	- method (int): method used to choose bin positioning (see bin_late_time)
	- earliest_tail_fit_start (float): earliest observations relative to peak to consider when fitting the tail
	- verdict_sigma (float): significance level for a bin to be considered a detection
	- tail_fit_chi2dof_tresh: chi2dof treshold for a successful tail fit
	- peak_mjd (float): found mjd of the SN peak
	- peak_mag (float): found mag of the SN peak
	- text (string): Notes on this object, will be saved in a .txt file
	- binlist (list): list of binning attempts, one for each filter, binsize, phase cobination
	- verdictlist (list): list of verdicts for each binning attempt in binlist
	- imp_df (DataFrame): df containing other values to save (peak_date, baseline_corrections)
	'''

	def __init__(self, args):
		#Unpack args into easier to use names
		self.name = args[0].name.rsplit('_S',1)[0]
		self.loc = args[0]
		self.refmags = args[1]
		self.saveloc = args[2] / self.name
		self.late_time = args[3]
		self.zp_rcid = args[4]
		self.cuts = args[5]
		self.base_gap = args[6]
		self.cols = args [7]
		self.binsizes = args[8]
		self.phases = args[9]
		self.method = args[10]
		self.earliest_tail_fit_start = args[11]
		self.verdict_sigma = args[12]
		self.tail_fit_chi2dof_tresh = args[13]
		self.text = 'Notes on ' + self.name + ':\n\n'
		#Initialize the lists that will store the bins & verdicts
		self.binlist = []
		self.verdictlist = []
		self.imp_df = pd.DataFrame(columns=['name', 'val'])
		#Make the directory to save things in
		self.saveloc.mkdir(exist_ok=True)
		#Load the lc
		self.load_source()
		return

	def load_source(self):
		#Load the lc
		try:
			self.lc = pd.read_csv(self.loc, usecols = self.cols, comment='#')
		except: #In case something goes wrong when reading in
			self.text += f'Could not read in lc for {self.name}\n'+ traceback.format_exc() + '\n'
			#print(self.name)
			self.peak_mjd = 0
			self.lc = pd.DataFrame(columns=self.cols)
			return
		#Rename problematic columns
		self.lc.rename(columns={'filter':'obs_filter', 'Fratio.err':'Fratio_err',
						   		'ampl.err':'ampl_err'}, inplace=True)
		#Add cloudy, cuts, & refmags
		self.calc_cloudy()
		self.apply_cuts()
		self.match_ref_im()
		#Find the peak of the SN
		self.find_peak_date()
		#Save this lc before bad points are removed
		nr_points = [len(self.lc[self.lc.obs_filter.str.contains('g')]),
					 len(self.lc[self.lc.obs_filter.str.contains('r')]),
					 len(self.lc[self.lc.obs_filter.str.contains('i')])]
		self.lc.to_csv(self.saveloc/'lc_orig.csv')
		self.imp_df = self.imp_df.append({'name':'peak_mjd_before_baseline', 'val':self.peak_mjd},
										 ignore_index=True)
		#apply baseline corrections
		self.correct_baseline()
		#Save the baseline corrected lc containing only useful points
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

	def calc_cloudy(self):
		#Cloudy has 4 conditions, pass 1 & the point should be removed
		#(Conditions from Adam Miller / Mat Smith)
	    #They get values 1,2,4,8 to keep track of which condition removed each point,
	    #& 16 for no_data (field=-99)
	    cloudy = np.zeros(len(self.lc))
	    #1) magzp > val I - val II * airmass      Wrong zp
	    cloudy[np.where(((self.lc.obs_filter.str.contains('g')) & \
	    				 (self.lc.magzp > 26.7-0.2*self.lc.airmass)) | \
	                    ((self.lc.obs_filter.str.contains('r')) & \
	                     (self.lc.magzp > 26.65-0.15*self.lc.airmass)) | \
	                    ((self.lc.obs_filter.str.contains('i')) & \
	                     (self.lc.magzp > 26.0-0.07*self.lc.airmass)))] += 1
	    #2) magzprms > val III                    Spread in zp too large
	    cloudy[np.where(((self.lc.obs_filter.str.contains('g')) & (self.lc.magzprms > 0.06)) | \
	                    ((self.lc.obs_filter.str.contains('r')) & (self.lc.magzprms > 0.05)) | \
	                    ((self.lc.obs_filter.str.contains('i')) & (self.lc.magzprms > 0.06)))] += 2
	    #3) nmatches < val IV                     Not enough catalog sources matched for alignment etc.
	    cloudy[np.where(((self.lc.obs_filter.str.contains('g')) & (self.lc.nmatches < 80)) | \
	                    ((self.lc.obs_filter.str.contains('r')) & (self.lc.nmatches < 120)) | \
	                    ((self.lc.obs_filter.str.contains('i')) & (self.lc.nmatches < 100)))] += 4
	    #4) magzp < zp_rcid - val II * airmass    Wrong zp
	    for _ in self.lc.rcid.unique():
	        cloudy[np.where(((self.lc.obs_filter.str.contains('g')) & (self.lc.rcid == _) & \
	        				 (self.lc.magzp < self.zp_rcid.g.iloc[int(_)]-0.2*self.lc.airmass)) | \
	                        ((self.lc.obs_filter.str.contains('r')) & (self.lc.rcid == _) & \
	                         (self.lc.magzp < self.zp_rcid.r.iloc[int(_)]-0.15*self.lc.airmass)) | \
	                        ((self.lc.obs_filter.str.contains('i')) & (self.lc.rcid == _) & \
	                         (self.lc.magzp < self.zp_rcid.i.iloc[int(_)]-0.07*self.lc.airmass)))] += 8
	    #5) field = -99                           No data
	    cloudy[np.where(self.lc.fieldid==-99)] += 16
	    self.lc['cloudy'] = cloudy
	    return

	def apply_cuts(self):
		cts = np.zeros(len(self.lc))
	    #cut 1: ampl_err > 2, removes failed photometry
		if 'ampl_err' in self.cuts.keys():
			cts[np.where(self.lc['ampl_err'] <= self.cuts['ampl_err'])] += 1
		#cut 2: chi2dof <= 3, removes bad psf fits
		if 'chi2dof' in self.cuts.keys():
			cts[np.where(self.lc.chi2dof > self.cuts['chi2dof'])] += 2
		#cut 3: seeing <= 3, removes bad seeing
		if 'seeing' in self.cuts.keys():
			cts[np.where(self.lc.seeing > self.cuts['seeing'])] += 4
		#cut 4: cloudy = False
		if 'cloudy' in self.cuts.keys():
			cts[np.where(self.lc.cloudy != self.cuts['cloudy'])] += 8
		#cut 5: infobits = 0 (or NaN)
		if 'infobits' in self.cuts.keys():
			#This keeps infobits=NaN in as well, NaNs should become values in the idr
	 		#(in August according to Mat)
			cts[np.where((self.lc.infobits > self.cuts['infobits']))] += 16
		self.lc['cuts'] = cts
		return

	def match_ref_im(self):
		'''
		Match and add the reference image magnitude limit to each observation.
		'''
		self.text += '-----\nmatch reference images\n\n'
		self.lc['ref_maglim'] = ''
		for _ in self.lc.index:
			if 'g' in self.lc.filename[_]:
				fid = 1
			elif 'r' in self.lc.filename[_]:
				fid = 2
			elif 'i' in self.lc.filename[_]:
				fid = 3
			else:
				self.text += 'ERROR: could not determine filter\n'
				continue
			self.lc.loc[_, 'ref_maglim'] = self.refmags[(
				(self.refmags.field==int(self.lc.filename[_].split('_')[2])) &
				(self.refmags.fid==fid) & (self.refmags.rcid==self.lc.rcid[_]))
				].maglimit.iloc[0]
		return

	def correct_baseline(self):
		self.text += '-----\ncorrect baseline \n\n'
		#Begin with dropping non_used data (dropped pionts are in lc_orig)
		self.lc = self.lc[self.lc.cuts==0].copy()
		for band in ['g', 'r', 'i']:
			for field in self.lc[self.lc.obs_filter.str.contains(band)].fieldid.unique():
				for rcid in self.lc[((self.lc.obs_filter.str.contains(band)) &
									 (self.lc.fieldid == field))].rcid.unique():
					#Assuming peak_mjd = correct, only use the points before base_gap days before the peak
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
						basel_err = np.sqrt(sum(weights* (nondets.Fratio - basel)**2 /
											(sum(weights)*len(nondets)-1)))
						self.lc.loc[self.lc[((self.lc.obs_filter.str.contains(band)) &
									(self.lc.fieldid == field) & (self.lc.rcid == rcid))].index,
									'Fratio'] -= basel
						self.lc.loc[self.lc[((self.lc.obs_filter.str.contains(band)) &
									(self.lc.fieldid == field) & (self.lc.rcid == rcid))].index,
									'Fratio_err'] = np.sqrt(self.lc[((self.lc.obs_filter.str.contains(band))&
																	 (self.lc.fieldid == field)&
																	 (self.lc.rcid == rcid))].Fratio_err**2 +
																	 basel_err**2)
						self.text += f'The combination of filter {band}, field {field}, and rcid {rcid} has baseline correction {basel} +- {basel_err}\n'
						self.imp_df = self.imp_df.append({'name':f'b{band}f{field}rcid{rcid}', 'val':basel},
														 ignore_index=True)
						self.imp_df = self.imp_df.append({'name':f'b{band}f{field}rcid{rcid}_err', 'val':basel_err},
														 ignore_index=True)
					else:
						self.text += f'The combination of filter {band}, field {field}, and rcid {rcid} only has {len(this_combo)} points available. These observations are removed\n'
						self.lc.drop(self.lc[((self.lc.obs_filter.str.contains(band))&
											  (self.lc.fieldid == field)&
											  (self.lc.rcid == rcid))].index, inplace=True)
		return

	def find_peak_date(self):
		'''
		Determine the date of peak light.
		Assume the highest Fratio observation is the peak, check for another
		observation with at least 1/2 times the peak flux within close_obs_size
		days. If none are found it is not considered real. Drop and try again.
		'''
		self.text += '-----\nfind peak date\n\n'
		data = self.lc[self.lc.cuts==0].copy()
		if data.empty: #If this is empty, then at most 1 datapoint --> useless
			self.text += 'Not enough points to get peak date & do binning!\n'
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
					if len(ref)==1:
						if ((ref.startmjd.values[0]<self.peak_mjd)&(ref.endmjd.values[0]>self.peak_mjd)):
							self.text +=f'WARNING: band {band[0]}, field {field}, rcid {rcid} might have the SN in the ref image\n'
					elif len(ref)>1:
						self.text +=f'WARNING: multiple instances of band {band[0]}, field {field}, rcid {rcid}, cannot determine if the SN might be inthe ref image\n'

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
			points = self.lc[((self.lc.obs_filter.str.contains(attempt.obs_filter[0][-1]))&
							  (self.lc.obsmjd>self.peak_mjd+self.earliest_tail_fit_start)&
							  (self.lc.obsmjd<self.peak_mjd+self.late_time))]
			#Give the verdict and append it to the list
			verdict, err = give_verdict(attempt, points, self.peak_mjd, self.verdict_sigma,
										self.tail_fit_chi2dof_tresh, self.saveloc)
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
	if len(obj_data.lc[obj_data.lc.obsmjd>=obj_data.peak_mjd+obj_data.late_time]) > 0:
		#Do the binning for each combination of filter, binsize & phase
		obj_data.bin_all()
		#Save the bins
		all_bins = pd.concat(obj_data.binlist, ignore_index=True)
		all_bins.to_csv(obj_data.saveloc / 'bins.csv', index=False)
		#Run the filtering program
		obj_data.check_bins()
		#Save the result of the filtering program
		all_verdicts = pd.concat(obj_data.verdictlist, ignore_index=True)
		all_verdicts.to_csv(obj_data.saveloc / 'verdicts.csv', index=False)
	else:
		obj_data.text += 'Unfortunately, there are no useful late-time observations available. Either they do not exist, or they have been cut for some reason\nBinning could not be performed\n'
	#Save the notes & imp_df
	obj_data.imp_df.to_csv(obj_data.saveloc / 'derived_values.csv', index=False)
	ftext = open(obj_data.saveloc/'notes.txt', 'w')
	ftext.write(obj_data.text)
	ftext.close()
	return

def give_final_verdicts(obj_list, host_dat, min_sep, min_successes):
	#Give a final verdict for each object & save them
	final_verdicts = pd.DataFrame()
	for _ in obj_list:
		final_verdicts = final_verdicts.append(final_verdict(_, host_dat[host_dat.ztfname==_.name],
															 min_sep, min_successes),
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
			std_dev_full = np.sqrt(std_dev**2 + np.median(10**(-0.4*thisbin.ref_maglim)/5)**2)
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

def give_verdict(bins, points, peak_mjd, sigma, chi2dof_tresh, saveloc):
	err = None
	fit = None
	max_val = None
	verdict = 0
	#Are there any bins?
	if len(bins) > 0:
		#Are there any bins that are detections?
		if bins.significance.max() < sigma:
			verdict += 1
		else:
			verdict += 2
			#Trusted detections require Fratio >0, significance >= sigma, & enough points
			pos_bins = bins[bins.Fratio>0]
			trusted_bins = pos_bins[((pos_bins.significance>=sigma) &
									 (pos_bins.nr_binned>=np.maximum(2,
								 									 2+4*(-2.5*np.log10(pos_bins.Fratio)-21))))]
			#Do the bins with detections contain enough points?
			if not trusted_bins.empty:
				verdict += 4
				#Is it a single bin or are there multiple ones?
				if len(trusted_bins) == 1:
					verdict += 8
				#If there are multiple bins, are they ajacent to one another?
				elif 1 in (trusted_bins.index[1:]-trusted_bins.index[:-1]):
					verdict += 16
					#Is the first bin a detection? (Only interested if there are ajacent dets)
					if trusted_bins.index[0] == bins.index[0]:
						verdict += 32
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
	#Put the results in a df, including the fit result if it was made
	if fit != None:
		#Save the model
		x = np.linspace(fit.params['t0'].value, fit.params['t0'].value+1000, 3000)
		fit_res = pd.DataFrame({'x':x, 'y':fit.eval(x=x)*max_val, 'dy':fit.eval_uncertainty(x=x)*max_val})
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
								   'verdict':[verdict]})
		else:
			result = pd.DataFrame({'band':[bins.obs_filter[0]], 'binsize':[bins.binsize[0]],
								   'phase':[bins.phase[0]], 'a_init':[fit.params['a'].init_value*max_val],
								   'a':[fit.params['a'].value*max_val], 'sigma_a':[' '],
								   't_half_init':[fit.params['tau'].init_value],
								   't_half':[fit.params['tau'].value], 'sigma_t_half':[' '],
								   't0':[fit.params['t0'].value], 'chi2':[fit.chisqr],
								   'chi2dof':[fit.redchi], 'AIC':[fit.aic], 'BIC':[fit.bic],
								   'verdict':[verdict]})
	else:
		result = pd.DataFrame({'band':[bins.obs_filter[0]], 'binsize':[bins.binsize[0]],
							   'phase':[bins.phase[0]], 'a_init':[' '], 'a':[' '], 'sigma_a':[' '],
							   't_half_init':[' '], 't_half':[' '], 'sigma_t_half':[' '],
							   't0':[' '], 'chi2':[' '], 'chi2dof':[' '], 'AIC':[' '],
							   'BIC':[' '], 'verdict':[verdict]})
	return result, err

def final_verdict(obj_loc, host_dat, min_sep, min_successes):
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
								if ((host_dat.separation.values[0]<min_sep) & (host_dat.separation.values[0]>-99)): #Is it too close to the host nucleus?
									too_nuc = True
								else:
									too_nuc = False
									succes_attempts = bel_verdicts[((bel_verdicts.verdict&64!=0) | (bel_verdicts.verdict==22))]
							else:
								normal = False
								#Failed & non normal Ia fits are interesting --> successfull attempts are those with verdict 22 or with_fits not containing 1024 or 4096
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
	df = pd.DataFrame({'name':obj_loc.name, 'bins':bins, 'detections':dets, 'lone_bins':lone,
					   'adjacent':adjacent, 'possible_tail':pos_tail, 'fit_always_fails':fit_failed,
					   'normal_tail':normal, 'too_nuclear':too_nuc, 'suc_g':suc_g, 'suc_r':suc_r,
					   'suc_i':suc_i, 'nr_suc_g':nr_g, 'nr_suc_r':nr_r, 'nr_suc_i':nr_i}, index=[0])
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
	sn_hosts['separation'] = SkyCoord(sn_hosts.sn_ra*u.deg, sn_hosts.sn_dec*u.deg,
	                                  frame='icrs').separation(SkyCoord(sn_hosts.host_ra*u.deg,
	                                                                    sn_hosts.host_dec*u.deg,
	                                                                    frame='icrs')).arcsecond
	#Those without a host location get separation=-99
	sn_hosts.loc[((sn_hosts.host_ra==270)&(sn_hosts.host_dec==-80)), 'separation'] =- 99
	#add the catalog magnitudes for the host
	host_data = pd.read_csv(dat_list, header=0, usecols=['ztfname', 'ra_gal',
														 'dec_gal', 'host_cat_mag',
														 'host_cat_mag_err'])
	for i in sn_hosts[sn_hosts.host_ra!=270].ztfname.values:
	    host_mags = eval(host_data[host_data.ztfname==i].host_cat_mag.values[0])
	    for key in host_mags:
	        sn_hosts.loc[sn_hosts[sn_hosts.ztfname==i].index, key] = host_mags[key]
	return sn_hosts

if (__name__ == "__main__"):
	main()
