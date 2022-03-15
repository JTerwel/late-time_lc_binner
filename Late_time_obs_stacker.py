'''
A program to stack and display late-time observations of the 2018 SN Ia dataset

Author: Jacco Terwel
Date: 26-04-21
'''

#Imports & global constants
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from os import getenv
from pathlib import Path
from astropy.time import Time
from scipy.optimize import curve_fit
from scipy.stats import t
from scipy.special import erf
from tqdm import tqdm

def main():
	'''
	Main function of the program.

	Make a list of locations to find the required data for each object, set
	the location where all results will be saved (separate folders for each
	object), and control the progress bar.
	'''
	#Set location where the results will be saved
	#saveloc = Path("/Users/terwelj/Projects/Late-time_signals/SN_Ia_new_stacker")			#SN Ia
	#saveloc = Path("/Users/terwelj/Projects/Late-time_signals/SN_Ia_sub_new_stacker")		#SN Ia sub
	#saveloc = Path("/Users/terwelj/Projects/Late-time_signals/version_test_results/sim_6")		#Testing
	#saveloc = Path("/Users/terwelj/Projects/testing_grounds")	#Single object
	saveloc = Path("/Users/terwelj/Projects/Late-time_signals/SN_Ia_2022lcs")				#All SN Ia, with updated lcs

	#Set the location of the object csv files and list them
	dataloc = getenv("ZTFDATA") + '/marshal'
	#datafiles = list(Path(dataloc, 'SN_Ia').rglob('*.csv'))				#SN Ia
	#datafiles = list(Path(dataloc, 'SN_Ia_sub').rglob('*.csv'))			#SN Ia sub
	#datafiles = [Path(dataloc, 'SN_Ia/ZTF18acrdwag_SNT_1e-08.csv'),
	#	Path(dataloc, 'SN_Ia/ZTF18abmxahs_SNT_1e-08.csv'),
	#	Path(dataloc, 'SN_Ia/ZTF18acurlbj_SNT_1e-08.csv'),
	#	Path(dataloc, 'SN_Ia/ZTF18acusrws_SNT_1e-08.csv'),
	#	Path(dataloc, 'SN_Ia/ZTF18aasdted_SNT_1e-08.csv'),
	#	Path(dataloc, 'SN_Ia/ZTF18aasprui_SNT_1e-08.csv'),
	#	Path(dataloc, 'SN_Ia/ZTF18aataafd_SNT_1e-08.csv'),
	#	Path(dataloc, 'SN_Ia/ZTF18acqqyah_SNT_1e-08.csv')]	#Testing
	#datafiles = list(Path('/Users/terwelj/Projects/Late-time_signals/sims/sim_6').rglob('sim*.csv')) #Testing
	#datafiles = [Path('/Users/terwelj/Projects/testing_grounds/ZTF21aagqcnl_SNT_5.000.csv')]	#Single object
	#print(datafiles)
	datafiles_all = list(Path("/Users/terwelj/Projects/Late-time_signals/ZTF18_Ia_sample_full+40_extra").rglob('*.csv'))
	obj_list = pd.read_csv('/Users/terwelj/Projects/Late-time_signals/ZTF18_Ia_names_10-02-2022.csv', header=None)
	datafiles = [i for i in datafiles_all if i.name.rsplit('_S',1)[0] in obj_list.values]
	print(len(datafiles)) #Only get the objects that are in my list as well as have lcs --> 952 objects

	#Set optional parameters & make the arg_list
	late_time = 100
	remove_sn_tails = True
	tail_removal_peak_tresh = 18
	make_plots = False
	args = [[f, saveloc, late_time, remove_sn_tails, tail_removal_peak_tresh,
		make_plots] for f in datafiles]

	#Make sure there isn't an overview.csv in saveloc before starting
	(saveloc / 'overview.csv').unlink(missing_ok=True)

	#use each cpu core separately & keep track of progress
	pool = mp.Pool(mp.cpu_count())
	list(tqdm(pool.imap_unordered(bin_late_time, args), total=len(datafiles)))
	pool.close()
	pool.join()
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
	g (DataFrame): filter_data object for the ZTF_g filter
	r (DataFrame): filter_data object for the ZTF_r filter
	i (DataFrame): filter_data object for the ZTF_i filter
	peak_mjd (float): SN peak light mjd
	peak_mag (float): SN peak light magnitude
	remove_sn_tails (bool): Fit & remove SN tail on bright SN?
	trpt (float): Tail removal peak brightness treshold
	'''

	def __init__(self, obj_path, saveloc, late_time, remove_sn_tails, trpt):
		'''
		Class constructor.

		Paramters:
		obj_path (Path): Location of the object csv file to be binned
		name (string): Name of the object
		saveloc (Path): Location to save the results
		late_time (int): Nr. of days after peak needed to be binned
		remove_sn_tails (bool): Fit & remove SN tail on bright SN?
		trpt (float): Tail removal peak brightness treshold
		'''
		#Save input
		self.obj_path = obj_path
		self.name = obj_path.name.rsplit('_S',1)[0]#[:-14]
		self.saveloc = saveloc / obj_path.name.rsplit('_S',1)[0]#[:-14]
		self.late_time = late_time
		self.remove_sn_tails = remove_sn_tails
		self.trpt = trpt
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
		self.g = filter_data(data[
			data.obs_filter.str.contains('g')].reset_index(drop=True), 'ZTF_g')
		self.r = filter_data(data[
			data.obs_filter.str.contains('r')].reset_index(drop=True), 'ZTF_r')
		self.i = filter_data(data[
			data.obs_filter.str.contains('i')].reset_index(drop=True), 'ZTF_i')

		#Mention if not everything could be sorted for some reason
		if (len(self.g.data)+len(self.r.data)+len(self.i.data)!=len(data)):
			print("\n{} rows could not be sorted in g,r,i filter\n".format(
				len(data)-len(self.g.data)-len(self.r.data)-len(self.i.data)))

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
		try:
			peak_light = data.obsmjd[data.Fratio.idxmax()]
		except:
			print(f'{self.name}: peak light could not be found, setting it at 58247.2 at mag 17.5 for now')
			self.peak_mjd = 58247.2
			self.peak_mag = 17.5
			return
		close_obs = data[abs(data.obsmjd-peak_light)<close_obs_size]
		while len(close_obs.Fratio[close_obs.Fratio>
				0.5*close_obs.Fratio.max()])<2:
			data = data.drop([data.Fratio.idxmax()], axis=0)
			peak_light = data.obsmjd[data.Fratio.idxmax()]
			close_obs = data[abs(data.obsmjd-peak_light)<close_obs_size]

			#Not enough points to determine peak realness? Use last guess
			if len(data)<2:
				print("{}: Could not verify peak_light, using last guess".format(
					self.name))
				break
		self.peak_mjd = peak_light
		self.peak_mag = data.mag[data.Fratio.idxmax()]
		return

class filter_data:
	'''
	Holds all data for a given filter

	Attributes:
	data (DataFrame): lc points
	filter_name (string): Name of the filter used
	fit_start (float): tail line fit starting date w.r.t. peak mjd
	fit_end (float): tail line fit end date w.r.t. peak mjd
	fit_zero (float): zeropoint w.r.t. which the fit is made, improves fit
	a (float): tail line fit slope
	b (float): tail line fit intersect
	cov (matrix): covariance matrix of the fit
	chi2red (float): reduced chi square of the fit
	dof (float): degrees of freedom of the fit
	results (list): list of bin_results objects
	'''

	def __init__(self, data, fname):
		'''
		Class constructor

		Parameters:
		data (DataFrame): lc points
		fname (string): filter name
		'''
		self.data = data
		self.filter_name = fname
		self.fit_start = 80
		self.fit_end = 200
		self.fit_zero = self.fit_end - self.fit_start
		self.a = 0
		self.b = 0
		self.cov = np.zeros(shape=(2,2))
		self.chi2red = 0
		self.dof = 0
		self.results = []

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
	def __init__(self, data, late_start, binsize, start_phase, method=1):
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
		self.obs_filter = data.filter_name
		self.binsize = binsize
		self.phase = start_phase
		if method not in [1, 2, 3, 4]:
			print('Method not recognized: {}\nUsing method 1'.format(method))
			method = 1
		self.method = method
		self.result = self.bin_late_time(data.data)
		return

	def bin_late_time(self, data):
		'''
		Bin the given data starting from the start mjd in binsize bins.

		Parameters:
		data (DataFrame): Data to bin

		Returns:
		result (DataFrame): the resulting bins
		'''
		#Initialize DataFrame, bin counter, & 1st bin left side
		result = pd.DataFrame(columns=['obs_filter', 'binsize', 'phase',
			'method', 'mjd_start', 'mjd_stop', 'mjd_bin', 'Fratio', 'Fratio_err',
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
				mjd_bin = sum(thisbin.obsmjd*weights)/sum(weights)
				if len(thisbin)==1:
					std_dev = thisbin.Fratio_err.max()
				else:
					std_dev = np.sqrt(sum(weights * (thisbin.Fratio-fratio)**2)
									  / (sum(weights) * (len(thisbin)-1)))
				if std_dev == 0:		#If this happens, don't trust bin
					signif= 0
				else:
					signif = fratio/std_dev

				#Store in the result & update the counter
				result.loc[counter] = [self.obs_filter, self.binsize,
					self.phase, self.method, newbin_start, newbin_stop, mjd_bin, 
					fratio, fratio_err, std_dev, len(thisbin), signif]
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

def bin_late_time(args):
	'''
	Master function for binning late time observations.

	Parameters:
	args(list): list of the arguments, contains the following:
	obj (Path): Location of the object csv file to be binned
	saveloc (Path): Location to save the results
	late_time (int): Nr. of days after peak needed to be binned, default 100
	remove_sn_tails (bool): Fit & remove SN tail on bright SN? default False
	trpt (float): If peak brightness < trpt, no tail is removed, default 18
	make_plots (bool): make plots of the resulting bins, default False

	Returns:
	'''
	#Unpack args & load object
	obj_path, saveloc, late_time, remove_sn_tails, trpt, make_plots = args
	obj_data = photom_obj(obj_path, saveloc, late_time, remove_sn_tails, trpt)

	#Fit end of SN tail & remove it from interfering with the late time binning
	if ((obj_data.peak_mag < obj_data.trpt) & (obj_data.remove_sn_tails)):
		fit_sn_tails(obj_data)
		rm_fitted_tail(obj_data)
		save_tails(obj_data)
	else: #If no tails were removed, ensure this is reflected in the settings
		obj_data.remove_sn_tails = False

	#Plot original lc in each filter with fit in mag space
	if make_plots:
		plot_lc(obj_data)

	#Start of the binning procedure, for each filter use different bin sizes &
	#different start phases: 1st bin at late_time - size * phase days after peak
	sizes = [100, 75, 50, 25]
	phases = [0, 0.25, 0.5, 0.75]
	for size in sizes:
		for phase in phases:
			obj_data.g.results.append(bin_results(obj_data.g,
				obj_data.peak_mjd+obj_data.late_time, size, phase, method=4))
			obj_data.r.results.append(bin_results(obj_data.r,
				obj_data.peak_mjd+obj_data.late_time, size, phase, method=4))
			obj_data.i.results.append(bin_results(obj_data.i,
				obj_data.peak_mjd+obj_data.late_time, size, phase, method=4))

	#Plot bin results
	if make_plots:
		zpg = 4.880e7
		zpr = 2.708e7
		zpi = 4.880e7
		plot_bin_results(obj_data.g.data, obj_data.g.results, obj_data.peak_mjd,
			obj_data.late_time, sizes, phases, 'ZTF_g', zpg, obj_data.saveloc)
		plot_bin_results(obj_data.r.data, obj_data.r.results, obj_data.peak_mjd,
			obj_data.late_time, sizes, phases, 'ZTF_r', zpr, obj_data.saveloc)
		plot_bin_results(obj_data.i.data, obj_data.i.results, obj_data.peak_mjd,
			obj_data.late_time, sizes, phases, 'ZTF_i', zpi, obj_data.saveloc)
	
	#Save results & select + record most promising bins
	save_bins(obj_data.g.results, obj_data.r.results, obj_data.i.results,
		sizes, obj_data.saveloc)
	save_settings(obj_data)

	return

def save_bins(g_bins, r_bins, i_bins, binsizes, saveloc):
	'''
	Save the binning result for each filter, binsize, & phase

	All bins are put into 1 dataframe.
	They can be separated at a later point by the combination of their values
	in the first 3 colums (obs_mjd, binsize, filter) if so desired.

	#Also find and save the most promising bins in the overview file

	Parameters:
	g_bins, r_bins, i_bins(list): list of bin_results objects for g, r, i filter
	binsizes (list): binsizes used
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
	#for i in all_bins.obs_filter.unique():
	for i in ['ZTF_g', 'ZTF_r', 'ZTF_i']:
		for j in binsizes:
			overview_cols += [k + '_{}_bin{}'.format(i, j) for k in vals_to_save]
			this_set = all_bins[(all_bins.obs_filter==i) & (all_bins.binsize==j)]
			if this_set.empty: #No bins with this filter and binsize exist
				overview_vals += [0,0,0,0,0,0,0,0]
			else:
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
		print('Warning: Number of bin_results does not match all size, phase '\
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
	plt.cla()
	plt.clf()
	plt.close()
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
	Make sure fitted tail extends down to set maglim(e.g. all points were used)

	Parameters:
	obj_data (photom_obj): object to fit
	'''
	for obj in [obj_data.g, obj_data.r, obj_data.i]:
		counter = 0		#Avoid infinite looping
		while counter < 10:
			#Find points to use in the fit
			obj.fit_zero = obj_data.peak_mjd + (obj.fit_end+obj.fit_start)/2
			data_to_fit = obj.data[
				(obj.data.obsmjd>obj_data.peak_mjd+obj.fit_start) &
				(obj.data.obsmjd<obj_data.peak_mjd+obj.fit_end) &
				(obj.data.mag<99)]

			#Fit tail if it has at least 5 points, else go with last estimate
			if len(data_to_fit) > 4:
				[obj.a, obj.b], obj.cov = curve_fit(poly,
					data_to_fit.obsmjd-obj.fit_zero, data_to_fit.mag,
					sigma=data_to_fit.mag_err)
				obj.chi2red = sum(
					(poly(data_to_fit.obsmjd-obj.fit_zero, obj.a, obj.b)\
					-data_to_fit.mag)**2/data_to_fit.mag_err)\
					/ (len(data_to_fit)-2)
				obj.dof = len(data_to_fit)-2
			else:
				break

			#Find where the model crosses the maglim
			mjdlim = ((22-obj.b) / obj.a) + obj.fit_zero

			#Are there observations between the current & new fit_end mjd?
			if len(obj.data.obsmjd[
					((obj.data.obsmjd>obj_data.peak_mjd+obj.fit_end) &
					(obj.data.obsmjd<mjdlim)) | ((obj.data.obsmjd>mjdlim) &
					(obj.data.obsmjd<obj_data.peak_mjd+obj.fit_end))]) != 0:
				#y: update fit_end & counter
				obj.fit_end = mjdlim - obj_data.peak_mjd
				counter += 1
			else:
				#found the final model, break out of the while loop
				break
	return

def rm_fitted_tail(obj_data):
	'''
	Remove the SN tails by subtracting the fitted line in flux space.
	line fit params are saved in the provided class

	Parameters:
	obj_data (photom_obj): object to fit
	'''
	for obj in [obj_data.g, obj_data.r, obj_data.i]:
		indices, mod_y, mod_dy = frat_to_remove(obj, obj_data.peak_mjd)
		obj.data.loc[indices, 'Fratio'] -= mod_y
		obj.data.loc[indices, 'Fratio_err'] = np.sqrt(mod_dy**2 +
			obj.data.loc[indices, 'Fratio_err'].values**2)
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
	i=0
	for obj in [obj_data.g, obj_data.r, obj_data.i]:
		data.loc[i] = [obj.filter_name, obj.a, obj.b, obj.cov[0,0],
			obj.cov[0,1], obj.cov[1,0], obj.cov[1,1], obj.chi2red, obj.dof,
			obj.fit_start+obj_data.peak_mjd, obj.fit_end+obj_data.peak_mjd,
			obj.fit_zero]
		i += 1

	#Save DataFrame & return
	data.to_csv(obj_data.saveloc/'tail_fits.csv', index=False)
	return

def frat_to_remove(obj, peak_mjd):
	'''
	Calculate the fratio values that have to be removed

	Paramters:
	obj(filter_data): data from which the tail is removed
	peak_mjd (float): mjd of SN peak light

	Returns:
	indices (array): Indices of the observations to subtract
	mod_y (array): model Fratio
	mod_dy (array): error on mod_y
	'''
	#Select the region on which the model is fitted
	indices = obj.data[(obj.data.obsmjd>peak_mjd+obj.fit_start) &
		(obj.data.obsmjd<peak_mjd+obj.fit_end)].index
	dates = obj.data.obsmjd.loc[indices]

	#Calc the model values in the selected region
	mod_y = poly(dates-obj.fit_zero, obj.a, obj.b).values

	#Calculate corresponding errors in mag space
	mod_dy = condfidence_band(dates-obj.fit_zero, obj.cov, obj.chi2red, obj.dof,
		1).values
	
	#Convert to Fratio before returning
	if ((obj.a==0) & (obj.b==0)): #No fit was actually performed, mod_y & mod_dy are 0
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
	fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
	to_plot = [obj_data.g, obj_data.r, obj_data.i]
	fmts = ['.g', '.r', '.y']

	axs[2].set_xlabel('Date')
	for i in range(3):
		axs[i].set_ylabel('Magnitude')
		axs[i].set_ylim(23, 15)
		axs[i].set_title(to_plot[i].filter_name)
		axs[i].grid(color='gray', alpha=0.4)

		#Plot the lc data, peak date & start of late time observations
		axs[i].errorbar(to_plot[i].data.obsmjd[to_plot[i].data.mag<99],
			to_plot[i].data.mag[to_plot[i].data.mag<99],
			yerr=to_plot[i].data.mag_err[to_plot[i].data.mag<99], fmt=fmts[i])
		axs[i].scatter(to_plot[i].data.obsmjd[to_plot[i].data.mag==99],
			to_plot[i].data.upper_limit[to_plot[i].data.mag==99], marker='v',
			color='k')
		axs[i].axvline(obj_data.peak_mjd, ls='--', color='gray', alpha=0.6)
		axs[i].axvline(obj_data.peak_mjd+obj_data.late_time, ls='--',
			color='gray', alpha=0.6)

		#Plot line fit if it is in there
		if ((obj_data.peak_mag < obj_data.trpt)&(obj_data.remove_sn_tails)):
			fit_x = np.linspace(to_plot[i].fit_start+obj_data.peak_mjd,
				to_plot[i].fit_end+obj_data.peak_mjd, 100)
			fit_y = poly(fit_x-to_plot[i].fit_zero, to_plot[i].a, to_plot[i].b)
			fit_dy = condfidence_band(fit_x-to_plot[i].fit_zero, to_plot[i].cov,
				to_plot[i].chi2red, to_plot[i].dof, 3)
			axs[i].plot(fit_x, fit_y, 'm')
			axs[i].fill_between(fit_x, fit_y-fit_dy, fit_y+fit_dy, color='gray',
				alpha=0.6)

	#Save figure
	fig.tight_layout()
	fig.savefig(obj_data.saveloc / 'orig_lc.png')
	plt.cla()
	plt.clf()
	plt.close()
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
