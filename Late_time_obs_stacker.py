'''
A program to stack and display late-time observations of the 2018 SN Ia dataset

Author: Jacco Terwel
Date: 26-04-21
'''

#Imports & global constants
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import getenv
from pathlib import Path
from astropy.time import Time
from scipy.optimize import curve_fit
from scipy.stats import t
from scipy.special import erf

def main():
	'''
	Main function of the program.

	Make a list of locations to find the required data for each object, set
	the location where all results will be saved (separate folders for each
	object), and control the progress bar.
	'''
	#Set location where the results will be saved
	#saveloc = Path("/Users/terwelj/Projects/Late-time_signals/SN_Ia")			#SN Ia
	#saveloc = Path("/Users/terwelj/Projects/Late-time_signals/SN_Ia_sub")		#SN Ia sub
	saveloc = Path("/Users/terwelj/Projects/Late-time_signals/version_test_results")	#Testing

	#Set the location of the object csv files and list them
	dataloc = getenv("ZTFDATA") + '/marshal'
	#datafiles = list(Path(dataloc, 'SN_Ia').rglob('*.csv'))				#SN Ia
	#datafiles = list(Path(dataloc, 'SN_Ia_sub').rglob('*.csv'))			#SN Ia sub
	datafiles = [Path(dataloc, 'SN_Ia/ZTF18acrdwag_SNT_1e-08.csv'),
		Path(dataloc, 'SN_Ia/ZTF18abmxahs_SNT_1e-08.csv'),
		Path(dataloc, 'SN_Ia/ZTF18acurlbj_SNT_1e-08.csv'),
		Path(dataloc, 'SN_Ia/ZTF18acusrws_SNT_1e-08.csv'),
		Path(dataloc, 'SN_Ia/ZTF18acqqyah_SNT_1e-08.csv')]	#Testing

	#Set optional parameters
	#late_time = 100
	remove_sn_tails = True
	#tail_removal_peak_tresh = 18

	#Initiate progress bar
	nr_files = len(datafiles)
	i = 0

	#Make sure there sin't an overview.csv in saveloc before starting
	(saveloc / 'overview.csv').unlink(missing_ok=True)

	for f in datafiles: #Loop over all files
		#Update progress bar
		i += 1
		progress_bar = (int(np.floor(40*i/nr_files)))
		print(" "*50 + "Working on " + f.name[:-4] + " | " + str(i) + "/"\
			+ str(nr_files) + " | Total progress: [" + "%"*progress_bar\
			 + " "*(40-progress_bar) + "]", end='\r')
		
		#Do stuff (unf)
		bin_late_time(f, saveloc, remove_sn_tails=remove_sn_tails)

	print("\nDone")
	return


#*---------*
#| Classes |
#*---------*

class photom_obj:
	'''
	Holds all relevant information about the current object.

	Attributes:
	obj (Path): Location of the object csv file to be binned
	saveloc (Path): Location to save the results (new folder is made if needed)
	late_time (int): Nr. of days after peak needed to be binned
	g (DataFrame): g filter lc points
	r (DataFrame): r filter lc points
	i (DataFrame): i filter lc points
	peak_mjd (float): SN peak light mjd
	peak_mag (float): SN peak light magnitude
	remove_sn_tails (bool): Fit & remove SN tail on bright SN?
	trpt (float): Tail removal peak brightness treshold
	fit_start_date (float): date relative to peak of start of SN tail fit
	fit_end_date (float): date relative to peak of end of SN tail fit
	g_a, r_a, i_a (float): slope value of line fit for each filter
	g_b, r_b, i_b (float): intercect value of line fit for each filter
	g_cov, r_cov, i_cov (matrix): Covariance matrices of the line fits
	g_chi2red, r_chi2red, i_chi2red (float): reduced chi2 for the line fits
	g_dof, r_dof, i_dof (float): degrees of freedom for the line fits
	g_results, r_results, i_results (list): list of bin_results objects
	'''

	def __init__(self, obj_path, saveloc, late_time, remove_sn_tails, trpt):
		'''
		Class constructor.

		Paramters:
		obj_path (Path): Location of the object csv file to be binned
		saveloc (Path): Location to save the results
		late_time (int): Nr. of days after peak needed to be binned
		remove_sn_tails (bool): Fit & remove SN tail on bright SN?
		trpt (float): Tail removal peak brightness treshold
		'''
		#Save input
		self.obj_path = obj_path
		self.saveloc = saveloc / obj_path.name[:-4]
		self.late_time = late_time
		self.remove_sn_tails = remove_sn_tails
		self.trpt = trpt
		#Save non input
		self.fit_start_date = 50			#Need to play a bit more with these values
		self.fit_end_date = 200
		self.g_a = 0
		self.g_b = 0
		self.g_cov = np.zeros(shape=(2,2))
		self.g_chi2red = 0
		self.g_dof = 0
		self.r_a = 0
		self.r_b = 0
		self.r_cov = np.zeros(shape=(2,2))
		self.r_chi2red = 0
		self.r_dof = 0
		self.i_a = 0
		self.i_b = 0
		self.i_cov = np.zeros(shape=(2,2))
		self.i_chi2red = 0
		self.i_dof = 0
		self.g_results = []
		self.r_results = []
		self.i_results = []
		#Make sure the location exists & load the lc
		self.saveloc.mkdir(exist_ok=True)
		self.load_source()

	def load_source(self):
		'''
		Load object, remove data containing NaN, split into filters, and find
		peak light date.
		'''
		data = pd.read_csv(self.obj_path, header=0, usecols=['obsmjd',
			'filter', 'Fratio', 'Fratio.err', 'mag', 'mag_err', 'upper_limit', 
			'data_hasnan'])

		#Rename columns with problematic names & remove data with NaN
		data.rename(columns={'filter':'obs_filter', 'Fratio.err':'Fratio_err'},
			inplace=True)
		data = data[data.data_hasnan==False]
		
		#Separate filters
		self.g = data[data.obs_filter.str.contains('g')].reset_index(drop=True)
		self.r = data[data.obs_filter.str.contains('r')].reset_index(drop=True)
		self.i = data[data.obs_filter.str.contains('i')].reset_index(drop=True)

		#Mention if not everything could be sorted for some reason
		if (len(self.g)+len(self.r)+len(self.i)!=len(data)):
			print("\n{} rows could not be sorted in g,r,i filter\n".format(
				len(data)-len(self.g)-len(self.r)-len(self.i)))

		#Only use the best points to find the peak
		self.find_peak_date(data[data.Fratio_err<np.mean(data.Fratio_err)])
		return

	def find_peak_date(self, data):
		'''
		Determine the date of peak light.
		Assume the highest Fratio observation is the peak, check for another
		observation with at least 1/2 times the peak flux within close_obs_size
		days. If none are found it is not considered real. Drop and try again.

		Parameters:
		data (DataFrame): lc points to determine peak date of
		'''
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
				print("Could not verify peak_light, using last guess")
				break
		self.peak_mjd = peak_light
		self.peak_mag = data.mag[data.Fratio.idxmax()]
		return

class bin_results:
	'''
	Bin & save given data together with the arguments used in the call

	Attributes:
	late_start (float): MJD marking start of late time observations
	obs_filter (string): The filter in which the datapoints were observed
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
	result (DataFrame): The filled bins
	'''
	def __init__(self, data, late_start, obs_filter, binsize, start_phase, method=1):
		'''
		Class constructor:

		Parameters:
		data (DataFrame): Data to bin
		late_start (float): Only bin data after this date
		binsize (float): Size of the used bins
		start_phase (float): The of the 1st data point in the 1st bin
			(0 = left edge, 1 = right edge)
		method (int): How to place the bins (1st datapoint always in 1st bin)
		'''
		self.late_start = late_start
		self.obs_filter = obs_filter
		self.binsize = binsize
		self.phase = start_phase
		if method not in [1, 2, 3, 4]:
			print('\nMethod not recognized: {}\nUsing method 1'.format(method))
			method = 1
		self.method = method
		self.result = self.bin_late_time(data)
		return

	def bin_late_time(self, data):
		'''
		Bin the given data starting from the start mjd in binsize bins.

		Parameters:
		data (DataFrame): Data to bin

		Returns:
		result (DataFrame): the resulting bins

		DataFrame contents:
			mjd_start, mjd_stop: left, right edge of the bin
			binsize: mjd_stop - mjd_start
			Fratio: weighted mean of the bins
			Fratio_err: weighted error of Fratio
			Fratio_std: binned points standard deviation (=err if nr_binned=1)
			nr_binned: Amount of datapionts in the bin
			significance: Ratio of bin mean and std_dev (<0 if Fratio <0)
		'''
		#Initialize DataFrame, bin counter, & 1st bin left side
		result = pd.DataFrame(columns=['obs_filter', 'binsize', 'phase',
			'method', 'mjd_start', 'mjd_stop', 'Fratio', 'Fratio_err',
			'Fratio_std', 'nr_binned', 'significance'])
		counter = 0
		newbin_start =  data.obsmjd[data.obsmjd>=self.late_start].min()\
			- self.binsize*self.phase

		#Fill the bins 1 by 1
		while newbin_start < data.obsmjd.max():
			#Calc right side of this bin according to the chosen method
			newbin_stop = newbin_start + self.binsize
			if (((self.method == 3) | (self.method == 4)) & (
					len(data[(data.obsmjd>=newbin_stop) & 
						(data.obsmjd<newbin_stop+0.1*self.binsize)])!=0) & (
					len(data[(data.obsmjd>=newbin_stop) & 
						(data.obsmjd<newbin_stop+0.1*self.binsize)])<3) & (
					len(data[(data.obsmjd>=newbin_stop+0.1*self.binsize) & 
						(data.obsmjd<newbin_stop+2*self.binsize)])==0)):
				newbin_stop = data.obsmjd[(data.obsmjd>=newbin_stop) & (
					data.obsmjd<newbin_stop+0.1*self.binsize)].max() + 1e-7

			#Select data
			thisbin = data[(data.obsmjd>=self.late_start) &
				(data.obsmjd>=newbin_start) & (data.obsmjd<newbin_stop)]

			#If the bin isn't empty, fill it
			if len(thisbin)!=0:
				#Calc bin params
				weights = 1./thisbin.Fratio_err**2
				fratio = sum(thisbin.Fratio*weights)/sum(weights)
				fratio_err = 1/np.sqrt(sum(weights))
				if len(thisbin)==1:
					std_dev = thisbin.Fratio_err.max()
				else:
					std_dev = thisbin.Fratio.std()
				if std_dev == 0:		#If this happens, don't trust bin
					signif= 0
				else:
					signif = fratio/std_dev

				#Store in the result & update the counter
				result.loc[counter] = [self.obs_filter, self.binsize,
					self.phase, self.method, newbin_start, newbin_stop, fratio,
					fratio_err, std_dev, len(thisbin), signif]
				counter += 1

			#Calc left side of next bin according to the chosen method
			if (self.method == 1) | (self.method == 3):
				newbin_start = newbin_stop

			elif (self.method == 2) | (self.method == 4):
				#If there are no more datapoints to bin, break out of the loop
				if len(data[data.obsmjd>=newbin_start+self.binsize]) == 0:
					break

				next_obs = data.obsmjd[
					data.obsmjd>=newbin_stop].min()
				if next_obs < newbin_start+2*self.binsize:
					newbin_start = next_obs
				else:
					newbin_start = next_obs - self.binsize*self.phase

		#return the result
		return result


#*----------------*
#| Main functions |
#*----------------*

def bin_late_time(obj_path, saveloc, late_time=100, remove_sn_tails=False,
		trpt=18):
	'''
	Master function for binning late time observations.

	Parameters:
	obj (Path): Location of the object csv file to be binned
	saveloc (Path): Location to save the results
	late_time (int): Nr. of days after peak needed to be binned, default 100
	remove_sn_tails (bool): Fit & remove SN tail on bright SN? default False
	trpt (float): If peak brightness < trpt, no tail is removed, default 18

	Returns:
	'''
	#Load object
	obj_data = photom_obj(obj_path, saveloc, late_time, remove_sn_tails, trpt)

	#Fit end of SN tail & remove it from interfering with the late time binning
	if ((obj_data.peak_mag < obj_data.trpt) & (obj_data.remove_sn_tails)):
		fit_sn_tails(obj_data)
		rm_fitted_tail(obj_data)
		save_tails(obj_data)
	else: #If no tails were removed, ensure this is reflected in the settings
		obj_data.remove_sn_tails = False

	#Plot original lc in each filter with fit in mag space
	plot_lc(obj_data)

	#Start of the binning procedure, for each filter use different bin sizes &
	#different start phases: 1st bin at late_time - size * phase days after peak
	sizes = [100, 75, 50, 25]
	phases = [0, 0.25, 0.5, 0.75]
	for size in sizes:
		for phase in phases:
			obj_data.g_results.append(bin_results(obj_data.g,
				obj_data.peak_mjd+obj_data.late_time, 'ZTF_g', size, phase,
				method=4))
			obj_data.r_results.append(bin_results(obj_data.r,
				obj_data.peak_mjd+obj_data.late_time, 'ZTF_r', size, phase,
				method=4))
			obj_data.i_results.append(bin_results(obj_data.i,
				obj_data.peak_mjd+obj_data.late_time, 'ZTF_i', size, phase,
				method=4))

	#Plot bin results
	zpg = 4.880e7
	zpr = 2.708e7
	zpi = 4.880e7
	plot_bin_results(obj_data.g, obj_data.g_results, obj_data.peak_mjd,
		obj_data.late_time, sizes, phases, 'ZTF_g', zpg, obj_data.saveloc)
	plot_bin_results(obj_data.r, obj_data.r_results, obj_data.peak_mjd,
		obj_data.late_time, sizes, phases, 'ZTF_r', zpr, obj_data.saveloc)
	plot_bin_results(obj_data.i, obj_data.i_results, obj_data.peak_mjd,
		obj_data.late_time, sizes, phases, 'ZTF_i', zpi, obj_data.saveloc)
	
	#Save results & select + record most promising bins
	save_bins(obj_data.g_results, obj_data.r_results, obj_data.i_results,
		obj_data.saveloc)
	save_settings(obj_data)

	return

def save_bins(g_bins, r_bins, i_bins, saveloc):
	'''
	Save the binning result for each filter, binsize, & phase

	All bins are put into 1 dataframe.
	They can be separated at a later point by the combination of their values
	in the first 3 colums (obs_mjd, binsize, filter) if so desired.

	#Also find and save the most promising bins in the overview file

	Parameters:
	g_bins, r_bins, i_bins(list): list of bin_results objects for g, r, i filter
	saveloc (Path): Location to save the result.
	'''
	df_list = [obj.result for obj in g_bins + r_bins + i_bins]
	all_bins = pd.concat(df_list, ignore_index=True)
	all_bins.to_csv(saveloc / 'all_bins.csv', index=False)

	#Find the most promising bins (if any) in every set of bins with the same
	#filter & binsize (significance > 0 required)
	vals_to_save = ['phase', 'mjd_start', 'mjd_stop', 'nr_binned', 'Fratio',
		'Fratio_err', 'Fratio_std', 'significance']
	overview_vals = [saveloc.name]
	overview_cols = ['obj_name']
	for i in all_bins.obs_filter.unique():
		for j in all_bins.binsize.unique():
			overview_cols += [k + '_{}_bin{}'.format(i, j) for k in vals_to_save]
			this_set = all_bins[(all_bins.obs_filter==i) & (all_bins.binsize==j)]
			best_bin = this_set[
				this_set.significance==this_set.significance.max()].iloc[0]
			if best_bin.significance > 0:
				overview_vals += [best_bin.phase, best_bin.mjd_start,
					best_bin.mjd_stop, best_bin.nr_binned, best_bin.Fratio,
					best_bin.Fratio_err, best_bin.Fratio_std,
					best_bin.significance]
			else: #Nothing interesting found
				overview_vals += [0,0,0,0,0,0,0,0]
	best_bins = pd.DataFrame(columns=overview_cols)
	best_bins.loc[0] = overview_vals

	#Add to overview file, or create new file with header if there is none
	with open(saveloc.parent / 'overview.csv', 'a') as f:
		best_bins.to_csv(f, header=(f.tell()==0), index=False)

	return

def save_settings(obj_data):
	'''
	Save the general settings & parameters used while binning this object

	Parameters:
	obj_data (photom_obj): object whose settings need to be saved
	'''
	set_data = [obj_data.obj_path, obj_data.peak_mjd, obj_data.peak_mag,
		obj_data.late_time, obj_data.remove_sn_tails, obj_data.trpt]
	set_idx = ['obj_path', 'peak_date', 'peak_mag', 'late_time',
		'remove_sn_tails', 'tail_removal_peak_treshold']
	settings = pd.Series(set_data, index=set_idx)
	settings.to_csv(obj_data.saveloc / 'settings.csv')
	return

def plot_bin_results(data, result_list, peak_mjd, late_time, sizes, phases,
		filt, zp, saveloc):
	'''
	Plot the result of binning the data in a single filter.

	Parameters:
	data (DataFrame): Datapoints that were binned
	result_list (list): List of bin_results objects
	peak_mjd (float): Date of SN peak
	late_time (float): Nr of days after the SN where datapoints are late_time
	sizes (list): List of used binsizes
	phases (list): List of used phases
	filt (string): Used filter
	zp (float): factor to convert given values to 10**-16 erg/s/cm**2/AA
	saveloc (Time): location to save the plot
	'''
	#First check if nr. bin_results objects = combinations of size & phase
	if len(result_list) != len(sizes) * len(phases):
		print('\nWarning: Number of bin_results does not match all size, phase '\
			'combinations!\nAs a result of this, some subplots may be empty. '\
			'(filter = '+filt+')')

	#Initialize plot and set axis names
	#Different sizes on horizontal axis, different phases on vertical axis
	fig, axs = plt.subplots(len(phases), len(sizes), figsize=(12, 12),
		sharex=True, sharey=True)
	for i in range(len(sizes)):
		axs[0][i].set_xlabel('binsize {}'.format(sizes[i]))
		axs[0][i].xaxis.set_label_position('top')
	for i in range(len(phases)):
		axs[i][-1].set_ylabel('phase {}'.format(phases[i]))
		axs[i][-1].yaxis.set_label_position('right')
	
	#Fake subplot to set shared labes correctly
	fake_plot = fig.add_subplot(111, frame_on=False)
	fake_plot.set_title('Bin results for {} filter'.format(filt), y=1.05)
	fake_plot.set_xlabel('Days after peak at MJD {}'.format(peak_mjd),
		labelpad=25)
	fake_plot.set_ylabel(r'Flux (10$^{-16}$ erg/s/cm$^2$/$\AA$)', labelpad=45)
	fake_plot.set_xticks([])
	fake_plot.set_yticks([])
	
	#Add a horizontal line at 0, vertical line at late_time, &
	#binned datapoints (+ unceretainties) in all plots
	for i in range(len(sizes)):
		for j in range(len(phases)):
			axs[j][i].axhline(0, color='k')
			axs[j][i].axvline(late_time, ls='--', color='gray')
			axs[j][i].errorbar(
				data[data.obsmjd>peak_mjd+late_time].obsmjd-peak_mjd,
				data[data.obsmjd>peak_mjd+late_time].Fratio*zp,
				yerr=data[data.obsmjd>peak_mjd+late_time].Fratio_err*zp,
				fmt='.', color='r')

	#For each bin_results object, find the correct subplot and plot the the
	#bins (+ uncertainties)
	for i in result_list:
		#Find correct subplot
		x = sizes.index(i.binsize)
		y = phases.index(i.phase)

		#Plot 1 bin at a time
		for j in range(len(i.result)):
			axs[y][x].plot(
				[i.result.mjd_start[j]-peak_mjd, i.result.mjd_stop[j]-peak_mjd],
				[i.result.Fratio[j]*zp, i.result.Fratio[j]*zp], 'b')
			axs[y][x].fill_between(
				[i.result.mjd_start[j]-peak_mjd, i.result.mjd_stop[j]-peak_mjd],
				[(i.result.Fratio[j]-i.result.Fratio_std[j])*zp,
					(i.result.Fratio[j]-i.result.Fratio_std[j])*zp],
				[(i.result.Fratio[j]+i.result.Fratio_std[j])*zp,
					(i.result.Fratio[j]+i.result.Fratio_std[j])*zp],
				color='b', alpha=0.3)
			axs[y][x].fill_between(
				[i.result.mjd_start[j]-peak_mjd, i.result.mjd_stop[j]-peak_mjd],
				[(i.result.Fratio[j]-i.result.Fratio_err[j])*zp,
					(i.result.Fratio[j]-i.result.Fratio_err[j])*zp],
				[(i.result.Fratio[j]+i.result.Fratio_err[j])*zp,
					(i.result.Fratio[j]+i.result.Fratio_err[j])*zp],
				color='k', alpha=0.3)

	#Save & return
	fig.tight_layout()
	fig.savefig(saveloc / (filt+'_binned.png'))
	return

def poly(x, a, b):
	'''
	Simple 1st order polynomial
	'''
	return a*x + b

def fit_sn_tails(obj_data):
	'''
	Fit the SN tails with a straight line in mag space.
	Only use points, no upper limits, and only if there are at least 5 points
	Results (a, b, cov, chi2red, dof) are saved in the provided class

	Parameters:
	obj_data (photom_obj): object to fit
	'''
	#Shift time axis such that 0 is in the middle of the fitted region
	shift = obj_data.peak_mjd + (obj_data.fit_end_date+obj_data.fit_start_date)/2
	#fit g
	data = obj_data.g[
		(obj_data.g.obsmjd>obj_data.peak_mjd+obj_data.fit_start_date) &
		(obj_data.g.obsmjd<obj_data.peak_mjd+obj_data.fit_end_date) &
		(obj_data.g.mag<99)]
	if len(data) > 4:
		fit, cov = curve_fit(poly, data.obsmjd-shift, data.mag,
			sigma=data.mag_err)
		obj_data.g_a = fit[0]
		obj_data.g_b = fit[1]
		obj_data.g_cov = cov
		obj_data.g_chi2red = sum(
			(poly(data.obsmjd-shift, fit[0], fit[1])-data.mag)**2/data.mag_err)\
			/ (len(data)-2)
		obj_data.g_dof = len(data)-2
	#fit r
	data = obj_data.r[
		(obj_data.r.obsmjd>obj_data.peak_mjd+obj_data.fit_start_date) &
		(obj_data.r.obsmjd<obj_data.peak_mjd+obj_data.fit_end_date) &
		(obj_data.r.mag<99)]
	if len(data) > 4:
		fit, cov = curve_fit(poly, data.obsmjd-shift, data.mag,
			sigma=data.mag_err)
		obj_data.r_a = fit[0]
		obj_data.r_b = fit[1]
		obj_data.r_cov = cov
		obj_data.r_chi2red = sum(
			(poly(data.obsmjd-shift, fit[0], fit[1])-data.mag)**2/data.mag_err)\
			/ (len(data)-2)
		obj_data.r_dof = len(data)-2
	#fit i
	data = obj_data.i[
		(obj_data.i.obsmjd>obj_data.peak_mjd+obj_data.fit_start_date) &
		(obj_data.i.obsmjd<obj_data.peak_mjd+obj_data.fit_end_date)&
		(obj_data.i.mag<99)]
	if len(data) > 4:
		fit, cov = curve_fit(poly, data.obsmjd-shift, data.mag,
			sigma=data.mag_err)
		obj_data.i_a = fit[0]
		obj_data.i_b = fit[1]
		obj_data.i_cov = cov
		obj_data.i_chi2red = sum(
			(poly(data.obsmjd-shift, fit[0], fit[1])-data.mag)**2/data.mag_err)\
			/ (len(data)-2)
		obj_data.i_dof = len(data)-2
	return

def rm_fitted_tail(obj_data):
	'''
	Remove the SN tails by subtracting the fitted line in flux space.
	line fit params are saved in the provided class

	Basic steps:
	-calc  model for all late time obs (+ errors)
	-rm obs where nothing will be subtracted (t>end date | mag>22)
	-record last date with subtractions
	-convert model mags to flux space (+ errors)
	-flux -= model flux (also update flux errors)

	Parameters:
	obj_data (photom_obj): object to fit
	'''
	#fit zeropoint
	fit_zero = obj_data.peak_mjd\
			+ (obj_data.fit_end_date+obj_data.fit_start_date)/2

	#Remove the model from the observations for each filter
	#g
	indices, mod_y, mod_dy = frat_to_remove(
		obj_data.g.obsmjd[obj_data.g.obsmjd>obj_data.peak_mjd+obj_data.late_time],
		obj_data.g_a, obj_data.g_b, obj_data.g_cov, obj_data.g_chi2red,
		obj_data.g_dof, fit_zero)
	obj_data.g.loc[indices, 'Fratio'] -= mod_y
	obj_data.g.loc[indices, 'Fratio_err'] = np.sqrt(
		obj_data.g.loc[indices, 'Fratio_err'].values**2 + mod_dy**2)
	#r
	indices, mod_y, mod_dy = frat_to_remove(
		obj_data.r.obsmjd[obj_data.r.obsmjd>obj_data.peak_mjd+obj_data.late_time],
		obj_data.r_a, obj_data.r_b, obj_data.r_cov, obj_data.r_chi2red,
		obj_data.r_dof, fit_zero)
	obj_data.r.loc[indices, 'Fratio'] -= mod_y
	obj_data.r.loc[indices, 'Fratio_err'] = np.sqrt(
		obj_data.r.loc[indices, 'Fratio_err']**2 + mod_dy**2)
	#i
	indices, mod_y, mod_dy = frat_to_remove(
		obj_data.i.obsmjd[obj_data.i.obsmjd>obj_data.peak_mjd+obj_data.late_time],
		obj_data.i_a, obj_data.i_b, obj_data.i_cov, obj_data.i_chi2red,
		obj_data.i_dof, fit_zero)
	obj_data.i.loc[indices, 'Fratio'] -= mod_y
	obj_data.i.loc[indices, 'Fratio_err'] = np.sqrt(
		obj_data.i.loc[indices, 'Fratio_err']**2 + mod_dy**2)
	return

def save_tails(obj_data):
	'''
	Save the details of the fitted tails

	Parameters:
	obj_data (photom_obj): Object containing the tails
	'''
	#Put all data in a DataFrame
	data = pd.DataFrame(columns=['obs_filter', 'a', 'b', 'Caa', 'Cab', 'Cba',
		'Cbb', 'chi2red', 'dof', 'fit_start_mjd', 'fit_end_mjd', 'fit_zero'])
	data.loc[0] = ['ZTF_g', obj_data.g_a, obj_data.g_b, obj_data.g_cov[0,0],
		obj_data.g_cov[0,1], obj_data.g_cov[1,0], obj_data.g_cov[1,1],
		obj_data.g_chi2red, obj_data.g_dof,
		obj_data.fit_start_date+obj_data.peak_mjd,
		obj_data.fit_end_date+obj_data.peak_mjd,
		obj_data.peak_mjd + (obj_data.fit_end_date+obj_data.fit_start_date)/2]
	data.loc[1] = ['ZTF_r', obj_data.r_a, obj_data.r_b, obj_data.r_cov[0,0],
		obj_data.r_cov[0,1], obj_data.r_cov[1,0], obj_data.r_cov[1,1],
		obj_data.r_chi2red, obj_data.r_dof,
		obj_data.fit_start_date+obj_data.peak_mjd,
		obj_data.fit_end_date+obj_data.peak_mjd,
		obj_data.peak_mjd + (obj_data.fit_end_date+obj_data.fit_start_date)/2]
	data.loc[2] = ['ZTF_i', obj_data.i_a, obj_data.i_b, obj_data.i_cov[0,0],
		obj_data.i_cov[0,1], obj_data.i_cov[1,0], obj_data.i_cov[1,1],
		obj_data.i_chi2red, obj_data.i_dof,
		obj_data.fit_start_date+obj_data.peak_mjd,
		obj_data.fit_end_date+obj_data.peak_mjd,
		obj_data.peak_mjd + (obj_data.fit_end_date+obj_data.fit_start_date)/2]

	#Save DataFrame & return
	data.to_csv(obj_data.saveloc/'tail_fits.csv', index=False)
	return

def frat_to_remove(dates, a, b, cov, chi2red, dof, fit_zero):
	'''
	Calculate the fratio values that have to be removed

	Paramters:
	dates (DataFrame): indexed obsmjd
	a (float): fit slope
	b (float): fit intersect
	cov (matrix): covariance matrix
	chi2red (float): reduced chi2
	dof (float): fit degrees of freedom
	fit_zero (float): fit zeropoint
	zp (float): flux zeropoint

	Returns:
	indices (array): Indices of the observations to subtract
	mod_y (array): model Fratio
	mod_dy (array): error on mod_y
	'''
	#Calc model y in mag space & select points where the subtraction will happen
	mod_y = poly(dates-fit_zero, a, b)
	indices = mod_y[mod_y<22].index
	mod_y = mod_y[indices].values

	#Calculate corresponding errors in mag space
	mod_dy = condfidence_band(dates[indices]-fit_zero, cov, chi2red, dof,
		1).values
	
	#Convert to Fratio before returning
	if ((a==0) & (b==0)): #No fit was actually performed, mod_y & mod_dy are 0
		mod_y = np.zeros_like(mod_y)
		mod_dy = np.zeros_like(mod_dy)
	else:
		mod_y = 10**(-mod_y/2.5)
		mod_dy = mod_y * np.log(10) * mod_dy / 2.5
	return indices, mod_y, mod_dy

def plot_lc(obj_data):
	'''
	Plot the lightcurve of the object showing the found peak date & start of
	late time observations. Also show the tail end fit if it was done.

	Parameters:
	obj_data (photom_obj): object to plot
	'''
	#Initiate plot, set ax labels & fig titles, and set ax limits
	fig, (ax_g, ax_r, ax_i) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
	ax_g.set_ylabel('Magnitude')
	ax_g.set_ylim(23, 15)
	ax_g.set_title('ZTF_g')
	ax_r.set_ylabel('Magnitude')
	ax_r.set_ylim(23, 15)
	ax_r.set_title('ZTF_r')
	ax_i.set_xlabel('Date')
	ax_i.set_ylabel('Magnitude')
	ax_i.set_ylim(23, 15)
	ax_i.set_title('ZTF_i')

	#Plot lc data, peak date & start of late time observations
	ax_g.errorbar(obj_data.g.obsmjd[obj_data.g.mag<99],
		obj_data.g.mag[obj_data.g.mag<99],
		yerr=obj_data.g.mag_err[obj_data.g.mag<99], fmt='.g')
	ax_g.scatter(obj_data.g.obsmjd[obj_data.g.mag==99],
		obj_data.g.upper_limit[obj_data.g.mag==99], marker='v', color='k')
	ax_g.axvline(obj_data.peak_mjd, ls='--', color='gray', alpha=0.6)
	ax_g.axvline(obj_data.peak_mjd+obj_data.late_time, ls='--', color='gray',
		alpha=0.6)

	ax_r.errorbar(obj_data.r.obsmjd[obj_data.r.mag<99],
		obj_data.r.mag[obj_data.r.mag<99],
		yerr=obj_data.r.mag_err[obj_data.r.mag<99], fmt='.r')
	ax_r.scatter(obj_data.r.obsmjd[obj_data.r.mag==99],
		obj_data.r.upper_limit[obj_data.r.mag==99], marker='v', color='k')
	ax_r.axvline(obj_data.peak_mjd, ls='--', color='gray', alpha=0.6)
	ax_r.axvline(obj_data.peak_mjd+obj_data.late_time, ls='--', color='gray',
		alpha=0.6)
	
	ax_i.errorbar(obj_data.i.obsmjd[obj_data.i.mag<99],
		obj_data.i.mag[obj_data.i.mag<99],
		yerr=obj_data.i.mag_err[obj_data.i.mag<99], fmt='.y')
	ax_i.scatter(obj_data.i.obsmjd[obj_data.i.mag==99],
		obj_data.i.upper_limit[obj_data.i.mag==99], marker='v', color='k')
	ax_i.axvline(obj_data.peak_mjd, ls='--', color='gray', alpha=0.6)
	ax_i.axvline(obj_data.peak_mjd+obj_data.late_time, ls='--', color='gray',
		alpha=0.6)

	#Plot line fit if it is there
	if ((obj_data.peak_mag < obj_data.trpt) & (obj_data.remove_sn_tails)):
		dates = obj_data.peak_mjd\
			+ np.array([obj_data.fit_start_date, obj_data.fit_end_date])
		fit_zero = obj_data.peak_mjd\
			+ (obj_data.fit_end_date+obj_data.fit_start_date)/2
		g_y = poly(dates-fit_zero, obj_data.g_a, obj_data.g_b)
		g_dy = condfidence_band(dates-fit_zero, obj_data.g_cov,
			obj_data.g_chi2red, obj_data.g_dof, 3)
		ax_g.plot(dates, g_y, 'm')
		ax_g.fill_between(dates, g_y-g_dy, g_y+g_dy, color='gray', alpha=0.6)
		r_y = poly(dates-fit_zero, obj_data.r_a, obj_data.r_b)
		r_dy = condfidence_band(dates-fit_zero, obj_data.r_cov,
			obj_data.r_chi2red, obj_data.r_dof, 3)
		ax_r.plot(dates, r_y, 'm')
		ax_r.fill_between(dates, r_y-r_dy, r_y+r_dy, color='gray', alpha=0.6)
		i_y = poly(dates-fit_zero, obj_data.i_a, obj_data.i_b)
		i_dy = condfidence_band(dates-fit_zero, obj_data.i_cov,
			obj_data.i_chi2red, obj_data.i_dof, 3)
		ax_i.plot(dates, i_y, 'm')
		ax_i.fill_between(dates, i_y-i_dy, i_y+i_dy, color='gray', alpha=0.6)

	#Save figure
	fig.tight_layout()
	fig.savefig(obj_data.saveloc / 'orig_lc.png')
	return

def condfidence_band(x, cov, chi2red, dof, sigma):
	'''
	Calculate the 3 sigma confidence band of the linear fit

	Parameters:
	x (array): Data used to fit
	cov (matrix): Covariance matrix
	chi2red (float): Reduced chi2 value
	dof (float): degrees of freedom in the fit
	sigma(float): size of confidence band

	Returns:
	dy (array): error on fitted y
	'''
	tval = t.ppf((erf(sigma/np.sqrt(2)) + 1)/2, dof)
	#line = ax+b --> dy/df0 = x, dy/df1 =1
	dy = tval * np.sqrt(chi2red\
		* (np.ones_like(x)*cov[1][1]+x*(cov[1][0]+cov[0][1])+x**2*cov[0][0]))
	return dy

if (__name__ == "__main__"):
	main()