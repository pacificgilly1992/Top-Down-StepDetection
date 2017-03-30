############################################################################
# Project: Electrical Pre-Conditioning of Convective Clouds,
# Title: Buffer the position when a Tip Bucket has tipped,
# Author: James Gilmore,
# Email: james.gilmore@pgr.reading.ac.uk.
# Version: 1.4.0
# Date: 06/03/17
# Status: Operational
# Change: Added 2D Top-Down
############################################################################
from __future__ import absolute_import, division, print_function
from array import array
import matplotlib.pyplot as plt
import numpy as np
import re,os,sys,inspect
import time as systime
import scipy as sc

sys.path.insert(0, '../../../GlobalFunctions')
from Gilly_Utilities import list_func


def Buffer_Type_Initaliser(rain=None, time=None, cal_no=None, cal_Tipmultipler=None, rain_offseter=None, PGRR_Key=None):
	"""Used to point to correct algorithm from a simple int value rather than rewriting 
	each methods definition. This will also estimate the number of Tips expected and 
	won't both proceding if not enough exist"""

	############################################################################
	"""If the number of potential tips are less than 5 then we can disregard as a data day"""
	raintip_est = Rain_Tip_Estimator(rain, cal_no, cal_Tipmultipler)								# Determine number of estimated steps to of occured
	print("Rain Tip Estimator:", raintip_est)
	if raintip_est <= 4: return [-1], [-1]

	############################################################################
	"""Find correct algorithm"""
		
	if inspect.isclass(PGRR_Key):
		if PGRR_Key.buff == 1:
			Tip_Time, raintot = Buffer_Tips_TopDown(rain, time, True)
		elif PGRR_Key.buff == 2:
			Tip_Time, raintot = Buffer_Tips_Threshold(rain, time, cal_no, cal_Tipmultipler)
		elif PGRR_Key.buff == 3:
			Tip_Time, raintot = Buffer_Tips_MeanGradient(rain, time, cal_no, cal_Tipmultipler, rain_offseter, 1)
		elif PGRR_Key.buff == 4:
			Tip_Time, raintot = Buffer_Tips_Converger(rain, time, cal_no, cal_Tipmultipler, 1)
		elif PGRR_Key.buff == 5:
			Tip_Time, raintot = Buffer_Tips_MovingAverage(rain, time, cal_no, cal_Tipmultipler, 30)
		elif PGRR_Key.buff == 6:
			Tip_Time, raintot = Buffer_Tips_BottomUp(rain, time, cal_no, cal_Tipmultipler)
		elif PGRR_Key.buff == 7:
			Tip_Time, raintot = Buffer_Tips_Mallat(rain, time)
		else:
			sys.exit("No Tip Buffer Mechanism was Chosen! v1")
	else:
		if PGRR_Key == 1:
			Tip_Time, raintot = Buffer_Tips_TopDown(rain, time, True)
			#Tip_Time, raintot = 1, 0
		elif PGRR_Key == 2:
			#Tip_Time, raintot = Buffer_Tips_Threshold(rain, time, cal_no, cal_Tipmultipler)
			Tip_Time, raintot = 0, 0
		elif PGRR_Key == 3:
			#Tip_Time, raintot = Buffer_Tips_MeanGradient(rain, time, cal_no, cal_Tipmultipler, rain_offseter, 1)
			Tip_Time, raintot = 0, 0
		elif PGRR_Key == 4:
			#Tip_Time, raintot = Buffer_Tips_Converger(rain, time, cal_no, cal_Tipmultipler, 1)
			Tip_Time, raintot = 0, 0
		elif PGRR_Key == 5:
			Tip_Time, raintot = Buffer_Tips_MovingAverage(rain, time, cal_no, cal_Tipmultipler, 30)
			#Tip_Time, raintot = 0, 0
		elif PGRR_Key == 6:
			#Tip_Time, raintot = Buffer_Tips_BottomUp(rain, time, cal_no, cal_Tipmultipler)
			Tip_Time, raintot = 0, 0
		elif PGRR_Key == 7:
			#Tip_Time, raintot = Buffer_Tips_Mallat(rain, time)
			Tip_Time, raintot = 0, 0
		else:
			sys.exit("No Tip Buffer Mechanism was Chosen! v2")
		
	return Tip_Time, raintot

def Rain_Tip_Estimator(rain, cal_no, cal_Tipmultipler):
	raintip_maxed = 0

	#Determine if the 5V maxima was reached and how many times
	if np.max(rain) > 4.98:
		for i in xrange(len(rain)-1):
			if rain[i+1]-rain[i] > 4.5:
				raintip_maxed +=1
	else:
		raintip_maxed = 0
	
	#As the instrument will sometimes reset itself prematuraley before the end of the day we set a large range to view the maxima
	#but we need to be careful not to cover the maxima of the instrument if that happens so we decrease the range until we find a
	#more sensible range
	rain_offseter = -10000
	try:
		while np.max(rain[rain_offseter:-1]) > 4.98:
			rain_offseter += 100
	except:
		rain_offseter = -1000
	raintip_est = int((5/(cal_Tipmultipler[cal_no]*0.2)*raintip_maxed)+round(np.max(rain[rain_offseter:-1])/(cal_Tipmultipler[cal_no]*0.2)))
	
	return raintip_est	
########################################################################################################################################################
"""ALL THE AVALIABLE ALGORITHMS ARE BELOW"""

############################################################################
"""Algorithm 1: The infamous Top-Down"""

def Buffer_Tips_BottomUp_v2(data, time, precision, threshold, output=True):
	Step_Total=0  
	rain_tip_binary_counter = int(len(data))
	Step_Time = np.array([], dtype=object)	
	Rain_Holder_Final = np.array([], dtype=float)
	rain_tip_finder_guide = np.ones(len(data))
	
	#Determine number of loops while data is still an integer number of itself
	it = 0
	n = rain_tip_binary_counter
	while int(n) == n: n /= 2; it+=1
	rain_tip_finder_all = zip(np.zeros([it]))
	
	for k in xrange(it):
		id = range(int(np.ceil(rain_tip_binary_counter)))
		rain_tip_finder = np.zeros(int(np.ceil(rain_tip_binary_counter)))
		rain_holder = np.zeros(int(np.ceil(rain_tip_binary_counter)))
		for i in xrange(int(np.ceil(rain_tip_binary_counter))):
			if rain_tip_finder_guide[i] == 1:
				rain_holder[i] = np.mean(data[((i+1)/int(np.ceil(rain_tip_binary_counter)))*len(data):((i+2)/int(np.ceil(rain_tip_binary_counter)))*len(data)])-np.mean(data[(i/int(np.ceil(rain_tip_binary_counter)))*len(data):((i+1)/int(np.ceil(rain_tip_binary_counter)))*len(data)])
		
		rain_holder_id = zip(id,rain_holder)
		
		searcher = 0
		for i in xrange(len(rain_holder_id)):
			if rain_holder_id[i][1] > threshold: #0.0103: #This is used to remove the doubles in the data, kinda of lag bustin coefficient
				rain_tip_finder[rain_holder_id[i][0]] = 1  #Lets search here again
				searcher+=1
		
		rain_tip_finder_all[k] = rain_tip_finder
		
		if output == True: print("Number of searchers:", searcher)
				
		#Update for next level of search
		rain_tip_binary_counter/=2
		rain_tip_finder_guide = np.zeros(int(np.ceil(rain_tip_binary_counter)))
		
		#Determines locations were to search next (i.e. if a previous level search didn't find any significant differences then why bother to continue to search)
		for i in xrange(1,len(rain_tip_finder)):
			if rain_tip_finder[i] == 1:
				rain_tip_finder_guide[np.floor(i/2)-1] = 1
				rain_tip_finder_guide[np.floor(i/2)] = 1
				rain_tip_finder_guide[np.floor(i/2)+1] = 1
				# if search_backward == True:
					# rain_tip_finder_guide[2*i-3] = 1
					# rain_tip_finder_guide[2*i-2] = 1
					# rain_tip_finder_guide[2*i-1] = 1
				# if search_forward == True:
					# rain_tip_finder_guide[2*i] = 1
					# rain_tip_finder_guide[2*i+1] = 1
					# rain_tip_finder_guide[2*i+2] = 1
					# rain_tip_finder_guide[2*i+3] = 1
		
		print("time[rain_tip_finder == 1]", int(np.ceil(rain_tip_binary_counter)))
		
	#Once we have leveled down to 1s resolution then we finish searching
	if output == True: print(int(np.ceil(rain_tip_binary_counter)))
	#for i in xrange(raintip_est): #This controls how many values you get out (i.e. the estimated number of tips)
	for i in xrange(int(np.ceil(rain_tip_binary_counter))):
		if rain_tip_finder[i] == 1: #At Top Level (i.e. rain_tip_finder_all[-1])
			# for j in xrange(it-1):	#Loop through each layer of rain_tip_finder to determine if there are any ones
				# if rain_tip_finder_all[-1-j,2*i] == 1:
					
				# if rain_tip_finder_all[-1-j,2*i-1] == 1:
			
			
			Step_Time = np.append(Step_Time, time[(rain_holder_id[i][0]/int(np.ceil(rain_tip_binary_counter)))*len(data)+1])
			
			
			Rain_Holder_Final = np.append(Rain_Holder_Final, rain_holder[i])
			Step_Total+=1
	kill = True	
	sys.exit()	
	
def Buffer_Tips_TopDown_OLD(rain, time, cal_no=None, cal_Tipmultipler=None):
	"""This method provides the best estimate of tip times when the noise of
	the instrument is significant enough to suppress their recognition. When
	the noise of the device is extremely minimal relatively then 
	'Buffer_Tips_Threshold' will probably provide better results. Currently
	this method produces ~90% of tips where the remaining 10% is obscured by
	false positives which have/have not been removed from end processing"""

	time[0]=time[-1]=0																# Set the first and last value of the time series equal to 0	
	raintot=0   																	# Number of tip buckets has occurred in a day
	rain_tip_binary_counter = int(np.round(len(rain)/2**10)) #int(2**np.ceil(np.log2(raintip_est)))
	#rain_tip_binary_counter = int(round(86400/2**7))
	Tip_Time = []
	rain_tip_finder_guide = np.zeros(np.round(len(rain)/2**10))
	#rain_tip_finder_guide = np.zeros(int(round(86400/2**7)))
	rain_tip_finder_guide[:] = 1
	
	#Buffer tip times
	#raintip_est = Rain_Tip_Estimator(rain, cal_no, cal_Tipmultipler)
	#print("Rain Tip Estimator:", raintip_est)
	
	#If the number of potential tips are less than 5 then we can disregard as a rain day
	#if raintip_est <= 4: return -1, -1
	kill = False
	#while raintip_est-raintot > 0:
	while kill == False:
		id = range(rain_tip_binary_counter)
		rain_tip_finder = np.zeros(rain_tip_binary_counter)
		rain_holder = np.zeros(rain_tip_binary_counter)
		for i in xrange(rain_tip_binary_counter):
			if rain_tip_finder_guide[i] == 1:
				rain_holder[i] = np.mean(rain[((i+1)/rain_tip_binary_counter)*len(rain):((i+2)/rain_tip_binary_counter)*len(rain)])-np.mean(rain[(i/rain_tip_binary_counter)*len(rain):((i+1)/rain_tip_binary_counter)*len(rain)])
		
		rain_holder_id = zip(id,rain_holder)
		#rain_holder_id = sorted(rain_holder_id,key=lambda l:l[1], reverse=True) #Use this to get best tips out rather than all detected tips
		
		#rain_holder_id = rain_holder_id[:raintip_est]
		
		searcher = 0
		for i in xrange(len(rain_holder_id)):
			if rain_holder_id[i][1] > 0.01: #0.0103: #This is used to remove the doubles in the data, kinda of lag bustin coefficient
				rain_tip_finder[rain_holder_id[i][0]] = 1  #Lets search here again
				searcher+=1
				
		print("Number of searchers:", searcher)
		#Once we have leveled down to 1s resolution then we finish searching
		if rain_tip_binary_counter == int(np.round(len(rain)/2**10)*2**10):
		#if rain_tip_binary_counter >= 86400:
			print(rain_tip_binary_counter)
			#for i in xrange(raintip_est): #This controls how many values you get out (i.e. the estimated number of tips)
			for i in xrange(rain_tip_binary_counter):
				if rain_tip_finder[i] == 1:
					Tip_Time.append(0.5*(time[(rain_holder_id[i][0]/rain_tip_binary_counter)*len(rain)]+time[((rain_holder_id[i][0]+1)/rain_tip_binary_counter)*len(rain)-1]))
					raintot+=1
			kill = True		
		
		#Update for next level of search
		rain_tip_binary_counter*=2
		rain_tip_finder_guide = np.zeros(rain_tip_binary_counter)
		
		#Determines locations were to search next (i.e. if a previous level search didn't find any significant differences then why bother to continue to search)
		for i in xrange(1,len(rain_tip_finder)):
			if rain_tip_finder[i] == 1:
				#rain_tip_finder_guide[2*i-1] = 1
				rain_tip_finder_guide[2*i] = 1
				rain_tip_finder_guide[2*i+1] = 1
				rain_tip_finder_guide[2*i+2] = 1
				rain_tip_finder_guide[2*i+3] = 1
	
	Tip_Time.sort()

	return Tip_Time, raintot
	
def Buffer_Tips_TopDown(data, time=None, output=True, precision=10, rain_thres=0.01, search_forward=True, search_backward=False):
	"""This method provides the best estimate of tip times when the noise of
	the instrument is significant enough to suppress their recognition. When
	the noise of the device is extremely minimal relatively then 
	'Buffer_Tips_Threshold' will probably provide better results. Currently
	this method produces ~95% of tips for moderate amount of noise where the 
	remaining 5% is obscured by false positives which have/have not been 
	removed from end processing.
	
	Parameters
    ----------
    data : numpy array
        1 dimensional data array that represents the data vales of the time 
		series
    time : numpy array, optional
        1 dimensional time series array attached to data stream to determine the timing a step
		was detected
	output : boolean, optional
		Outputs information to the user about the step detection at each iteration

    Returns
    -------
    Step_Time : numpy array
        The index or times of each detected step dependent on whether the 'time' 
		imput is given
	Step_Total : int
		Number of detected steps from the algorithm
	"""

	#Setting up initial variables and values
	#time[0]=time[-1]=0																				# Set the first and last value of the time series equal to 0	
	Step_Total=0   																					# Holds the total number of steps occuring in a day
	rain_tip_binary_counter = int(np.round(len(data)/2**precision)) #int(2**np.ceil(np.log2(raintip_est)))	# Inital number of search areas. Exponent determines how many levels
	#rain_tip_binary_counter = int(round(86400/2**7))
	Step_Time = np.array([], dtype=object)																					# Holds the index or time of each step detected)
	Rain_Holder_Final = np.array([], dtype = float)
	rain_tip_finder_guide = np.ones(np.round(len(data)/2**precision))										# Used to determine which bin to search. (1) Search; (0) Don't Search. (i.e. initially, since we haven't started searching yet we want to check all bins)
	#rain_tip_finder_guide = np.zeros(int(round(86400/2**7)))
	kill = False																					# Used to stop searching for steps and go to output
	
	#Buffer tip times	

	#while raintip_est-Step_Total > 0:
	while kill == False:
		id = range(rain_tip_binary_counter)
		rain_tip_finder = np.zeros(rain_tip_binary_counter)
		rain_holder = np.zeros(rain_tip_binary_counter)
		for i in xrange(rain_tip_binary_counter):
			if rain_tip_finder_guide[i] == 1:
				rain_holder[i] = np.mean(data[((i+1)/rain_tip_binary_counter)*len(data):((i+2)/rain_tip_binary_counter)*len(data)])-np.mean(data[(i/rain_tip_binary_counter)*len(data):((i+1)/rain_tip_binary_counter)*len(data)])
		
		rain_holder_id = zip(id,rain_holder)
		#rain_holder_id = sorted(rain_holder_id,key=lambda l:l[1], reverse=True) #Use this to get best tips out rather than all detected tips
		
		#rain_holder_id = rain_holder_id[:raintip_est]
		
		searcher = 0
		for i in xrange(len(rain_holder_id)):
			if rain_holder_id[i][1] > rain_thres: #0.0103: #This is used to remove the doubles in the data, kinda of lag bustin coefficient
				rain_tip_finder[rain_holder_id[i][0]] = 1  #Lets search here again
				searcher+=1
				
		if output == True: print("Number of searchers:", searcher)
		#Once we have leveled down to 1s resolution then we finish searching
		if rain_tip_binary_counter == int(np.round(len(data)/2**precision)*2**precision):
		#if rain_tip_binary_counter >= 86400:
			if output == True: print(rain_tip_binary_counter)
			#for i in xrange(raintip_est): #This controls how many values you get out (i.e. the estimated number of tips)
			for i in xrange(rain_tip_binary_counter):
				if rain_tip_finder[i] == 1:
					#Step_Time.append(0.5*(time[(rain_holder_id[i][0]/rain_tip_binary_counter)*len(data)+1]+time[((rain_holder_id[i][0]+1)/rain_tip_binary_counter)*len(data)]))
					Step_Time = np.append(Step_Time, time[(rain_holder_id[i][0]/rain_tip_binary_counter)*len(data)+1])
					Rain_Holder_Final = np.append(Rain_Holder_Final, rain_holder[i])
					Step_Total+=1
			kill = True	
		
		#Update for next level of search
		rain_tip_binary_counter*=2
		rain_tip_finder_guide = np.zeros(rain_tip_binary_counter)
		
		#Determines locations were to search next (i.e. if a previous level search didn't find any significant differences then why bother to continue to search)
		for i in xrange(1,len(rain_tip_finder)):
			if rain_tip_finder[i] == 1:
				if search_backward == True:
					rain_tip_finder_guide[2*i-3] = 1
					rain_tip_finder_guide[2*i-2] = 1
					rain_tip_finder_guide[2*i-1] = 1
				if search_forward == True:
					rain_tip_finder_guide[2*i] = 1
					rain_tip_finder_guide[2*i+1] = 1
					rain_tip_finder_guide[2*i+2] = 1
					rain_tip_finder_guide[2*i+3] = 1
	

	Step_Time = Step_Time[np.argsort(Step_Time)] #Sorts in ascending order the time of each step
	Rain_Holder_Final = Rain_Holder_Final[np.argsort(Step_Time)]
	#print(Step_Time, len(Step_Time))
	#print(len(Step_Time), Rain_Holder_Final)

	return Step_Time, Rain_Holder_Final

def Buffer_Tips_TopDown_v2(data, time=None, output=True, precision=10, rain_thres=0.01, search_forward=True, search_backward=False):
	"""This method provides the best estimate of tip times when the noise of
	the instrument is significant enough to suppress their recognition. When
	the noise of the device is extremely minimal relatively then 
	'Buffer_Tips_Threshold' will probably provide better results. Currently
	this method produces ~95% of tips for moderate amount of noise where the 
	remaining 5% is obscured by false positives which have/have not been 
	removed from end processing.
	
	Parameters
    ----------
    data : numpy array
        1 dimensional data array that represents the data vales of the time 
		series
    time : numpy array, optional
        1 dimensional time series array attached to data stream to determine the timing a step
		was detected
	output : boolean, optional
		Outputs information to the user about the step detection at each iteration

    Returns
    -------
    Step_Time : numpy array
        The index or times of each detected step dependent on whether the 'time' 
		imput is given
	Step_Total : int
		Number of detected steps from the algorithm
	"""

	#Setting up initial variables and values
	#time[0]=time[-1]=0									# Set the first and last value of the time series equal to 0	
	Step_Total=0   										# Holds the total number of steps occuring in a day
	
	#Get Multipler Sequence
	mul = get_multipler(len(data))
	print(mul)
	rain_tip_binary_counter = mul[0] 					# Inital number of search areas. Exponent determines how many levels
	Step_Time = np.array([], dtype=object)				# Holds the index or time of each step detected)
	Rain_Holder_Final = np.array([], dtype = float)
	rain_tip_finder_guide = np.ones(mul[0])				# Used to determine which bin to search. (1) Search; (0) Don't Search. (i.e. initially, since we haven't started searching yet we want to check all bins)
	kill = False										# Used to stop searching for steps and go to output
	
	# if rain_thres == 'auto':
		# rain_thres = np.median([np.std(data[i:i+1000], ddof=1) for i in xrange(int(len(data)/1000))])
		# #rain_thres = np.median(data[~np.isnan(data)])/np.std(data[~np.isnan(data)], ddof=1)
		# #rain_thres = np.median(data[~np.isnan(data)])/np.std(data[~np.isnan(data)], ddof=1)
		# print('rain_thres:', np.median([np.std(data[i:i+1000], ddof=1) for i in xrange(int(len(data)/1000))]))
		# print('rain_thres alt:', np.median(data[~np.isnan(data)])/np.std(data[~np.isnan(data)], ddof=1))
		
	#Buffer tip times	

	#while raintip_est-Step_Total > 0:
	for k in xrange(len(mul)+1):
		id = np.arange(rain_tip_binary_counter)
		rain_tip_finder = np.zeros(rain_tip_binary_counter)
		rain_holder = np.zeros(rain_tip_binary_counter)
		rain_holder_noise = np.zeros(rain_tip_binary_counter)
		
		#if rain_tip_finder_guide[0] == 1:
			# rain_holder_upper = data[(1/rain_tip_binary_counter)*len(data):(2/rain_tip_binary_counter)*len(data)]
			# rain_holder_lower = data[:(1/rain_tip_binary_counter)*len(data)]
			# rain_holder[0] = np.mean(rain_holder_upper[~np.isnan(rain_holder_upper)])-np.mean(rain_holder_lower[~np.isnan(rain_holder_lower)])
					
		for i in xrange(rain_tip_binary_counter-1):
			if rain_tip_finder_guide[i] == 1:
				
				rain_holder_upper = data[((i+1)/rain_tip_binary_counter)*len(data):((i+2)/rain_tip_binary_counter)*len(data)]
				rain_holder_lower = data[(i/rain_tip_binary_counter)*len(data):((i+1)/rain_tip_binary_counter)*len(data)]
				rain_holder_both = data[(i/rain_tip_binary_counter)*len(data):((i+2)/rain_tip_binary_counter)*len(data)]

				rain_holder[i] = np.mean(rain_holder_upper[~np.isnan(rain_holder_upper)])-np.mean(rain_holder_lower[~np.isnan(rain_holder_lower)])
				rain_holder_noise[i] = np.median(rain_holder_both[~np.isnan(rain_holder_both)])/np.std(rain_holder_both[~np.isnan(rain_holder_both)], ddof=1)
				#rain_holder_noise[i] = np.std(rain_holder_both[~np.isnan(rain_holder_both)], ddof=1)
				
		if rain_tip_finder_guide[-1] == 1:
			rain_holder_upper = data[((-1)/rain_tip_binary_counter)*len(data):]
			rain_holder_lower = data[(-2/rain_tip_binary_counter)*len(data):((-1)/rain_tip_binary_counter)*len(data)]
			rain_holder_both = data[(-2/rain_tip_binary_counter)*len(data):]
			rain_holder[-1] = np.mean(rain_holder_upper[~np.isnan(rain_holder_upper)])-np.mean(rain_holder_lower[~np.isnan(rain_holder_lower)])
			rain_holder_noise[-1] = np.median(rain_holder_both[~np.isnan(rain_holder_both)])/np.std(rain_holder_both[~np.isnan(rain_holder_both)], ddof=1)	
			#rain_holder_noise[-1] = np.std(rain_holder_both[~np.isnan(rain_holder_both)], ddof=1)	
			
		rain_holder_id = zip(id,rain_holder)
		
		#When we specifiy a noise level we build need to populate rain_holder_noise with these values
		if rain_thres != 'auto': rain_holder_noise = list_func([]).ljust(len(rain_holder_id), rain_thres, float)
				
		searcher = 0
		for i in xrange(len(rain_holder_id)):
			#print(rain_holder_id[i][1], ">", rain_holder_noise[i])
			if np.abs(rain_holder_id[i][1]) > rain_holder_noise[i]: #0.0103: #This is used to remove the doubles in the data, kinda of lag bustin coefficient
				rain_tip_finder[rain_holder_id[i][0]] = 1  #Lets search here again
				searcher+=1

		if output == True: print("Number of searchers:", searcher)
		
		if k == len(mul): continue #i.e. exit
		
		#Update for next level of search
		rain_tip_binary_counter*=mul[k]
		rain_tip_finder_guide = np.zeros(rain_tip_binary_counter)
		
		#Determines locations were to search next (i.e. if a previous level search didn't find any significant differences then why bother to continue to search)				
		for i in xrange(len(rain_tip_finder)-1):
			if rain_tip_finder[i] == 1:
				if search_backward == True:
					for j in xrange(mul[k]):
						rain_tip_finder_guide[mul[k]*i-(mul[k]+j-1)] = 1
				if search_forward == True:		
					for j in xrange(2*mul[k]):
						rain_tip_finder_guide[mul[k]*i+j] = 1

				# if search_backward == True:		
					# rain_tip_finder_guide[mul[k]*i-3] = 1
					# rain_tip_finder_guide[mul[k]*i-2] = 1
					# rain_tip_finder_guide[mul[k]*i-1] = 1
				# if search_forward == True:
					# rain_tip_finder_guide[mul[k]*i] = 1
					# rain_tip_finder_guide[mul[k]*i+1] = 1
					# rain_tip_finder_guide[mul[k]*i+2] = 1
					# rain_tip_finder_guide[mul[k]*i+3] = 1
		
		if search_backward == True:
			for j in xrange(mul[k]):
				rain_tip_finder_guide[mul[k]*(len(rain_tip_finder)-1)-(mul[k]+j-1)] = 1
		if search_forward == True:		
			for j in xrange(mul[k]):
				rain_tip_finder_guide[mul[k]*(len(rain_tip_finder)-1)+j] = 1
		
	#Once we have leveled down to 1s resolution then we finish searching
	if output == True: print(rain_tip_binary_counter, len(rain_tip_finder))
	#for i in xrange(raintip_est): #This controls how many values you get out (i.e. the estimated number of tips)
	
	for i in xrange(rain_tip_binary_counter):
		if rain_tip_finder[i] == 1:
			#Step_Time.append(0.5*(time[(rain_holder_id[i][0]/rain_tip_binary_counter)*len(data)+1]+time[((rain_holder_id[i][0]+1)/rain_tip_binary_counter)*len(data)]))
			try:
				Step_Time = np.append(Step_Time, time[rain_holder_id[i][0]+1])
			except IndexError:
				Step_Time = np.append(Step_Time, time[rain_holder_id[i][0]])
				
			Rain_Holder_Final = np.append(Rain_Holder_Final, rain_holder[i])
			Step_Total+=1
	kill = True					
					
	print('No. Tips:', len(rain_tip_finder[rain_tip_finder==1]))
	
	Step_Time = Step_Time[np.argsort(Step_Time)] #Sorts in ascending order the time of each step
	Rain_Holder_Final = Rain_Holder_Final[np.argsort(Step_Time)]
	#print(Step_Time, len(Step_Time))
	#print(len(Step_Time), Rain_Holder_Final)
	#sys.exit()
	return Step_Time, Rain_Holder_Final

def Buffer_Tips_TopDown_LeftRight(data, time=None, output=True, precision=10, rain_thres=0.01, search_forward=True, search_backward=False):
	"""This method provides the best estimate of tip times when the noise of
	the instrument is significant enough to suppress their recognition. When
	the noise of the device is extremely minimal relatively then 
	'Buffer_Tips_Threshold' will probably provide better results. Currently
	this method produces ~95% of tips for moderate amount of noise where the 
	remaining 5% is obscured by false positives which have/have not been 
	removed from end processing.
	
	Parameters
    ----------
    data : numpy array
        1 dimensional data array that represents the data vales of the time 
		series
    time : numpy array, optional
        1 dimensional time series array attached to data stream to determine the timing a step
		was detected
	output : boolean, optional
		Outputs information to the user about the step detection at each iteration

    Returns
    -------
    Step_Time : numpy array
        The index or times of each detected step dependent on whether the 'time' 
		imput is given
	Step_Total : int
		Number of detected steps from the algorithm
	"""

	#Setting up initial variables and values
	#time[0]=time[-1]=0																				# Set the first and last value of the time series equal to 0	
	Step_Total_Left = 0   																					# Holds the total number of steps occuring in a day
	Step_Total_Right = 0
	#Get Multipler Sequence
	mul = get_multipler(len(data))
	print(mul)
	data_rev = np.array(list(reversed(data)),dtype=float)

	rain_tip_binary_counter = mul[0] #int(2**np.ceil(np.log2(raintip_est)))	# Inital number of search areas. Exponent determines how many levels
	#rain_tip_binary_counter = int(round(86400/2**7))
	Step_Time_Left = np.array([], dtype=object)																					# Holds the index or time of each step detected)
	Step_Time_Right = np.array([], dtype=object)
	Rain_Holder_Final_Left = np.array([], dtype = float)
	Rain_Holder_Final_Right = np.array([], dtype = float)
	rain_tip_finder_guide_left = np.ones(mul[0])										# Used to determine which bin to search. (1) Search; (0) Don't Search. (i.e. initially, since we haven't started searching yet we want to check all bins)
	rain_tip_finder_guide_right = np.ones(mul[0])
	#rain_tip_finder_guide = np.zeros(int(round(86400/2**7)))
	kill = False																					# Used to stop searching for steps and go to output
	
	#Buffer tip times	

	#while raintip_est-Step_Total > 0:
	for k in xrange(len(mul)+1):
		id = range(rain_tip_binary_counter)
		rain_tip_finder_left = np.zeros(rain_tip_binary_counter)
		rain_tip_finder_right = np.zeros(rain_tip_binary_counter)
		rain_holder_left = np.zeros(rain_tip_binary_counter)
		rain_holder_right = np.zeros(rain_tip_binary_counter)
		
		#if rain_tip_finder_guide[0] == 1:
			# rain_holder_upper = data[(1/rain_tip_binary_counter)*len(data):(2/rain_tip_binary_counter)*len(data)]
			# rain_holder_lower = data[:(1/rain_tip_binary_counter)*len(data)]
			# rain_holder[0] = np.mean(rain_holder_upper[~np.isnan(rain_holder_upper)])-np.mean(rain_holder_lower[~np.isnan(rain_holder_lower)])
					
		for i in xrange(rain_tip_binary_counter-1):
			if rain_tip_finder_guide_left[i] == 1:
				
				rain_holder_left_upper = data[((i+1)/rain_tip_binary_counter)*len(data):((i+2)/rain_tip_binary_counter)*len(data)]
				rain_holder_left_lower = data[(i/rain_tip_binary_counter)*len(data):((i+1)/rain_tip_binary_counter)*len(data)]
				rain_holder_left[i] = np.mean(rain_holder_left_upper[~np.isnan(rain_holder_left_upper)])-np.mean(rain_holder_left_lower[~np.isnan(rain_holder_left_lower)])
		
			if rain_tip_finder_guide_right[i] == 1:
				rain_holder_right_upper = data_rev[((i+1)/rain_tip_binary_counter)*len(data):((i+2)/rain_tip_binary_counter)*len(data)]
				rain_holder_right_lower = data_rev[(i/rain_tip_binary_counter)*len(data):((i+1)/rain_tip_binary_counter)*len(data)]
				rain_holder_right[i] = np.mean(rain_holder_right_lower[~np.isnan(rain_holder_right_lower)])-np.mean(rain_holder_right_upper[~np.isnan(rain_holder_right_upper)])
				
		if rain_tip_finder_guide_left[-1] == 1:
			rain_holder_left_upper = data[((-1)/rain_tip_binary_counter)*len(data):]
			rain_holder_left_lower = data[(-2/rain_tip_binary_counter)*len(data):((-1)/rain_tip_binary_counter)*len(data)]
			rain_holder_left[-1] = np.mean(rain_holder_left_upper[~np.isnan(rain_holder_left_upper)])-np.mean(rain_holder_left_lower[~np.isnan(rain_holder_left_lower)])
		
		if rain_tip_finder_guide_right[-1] == 1:
			rain_holder_right_upper = data_rev[(-2/rain_tip_binary_counter)*len(data_rev):((-1)/rain_tip_binary_counter)*len(data_rev)]
			rain_holder_right_lower = data_rev[((-1)/rain_tip_binary_counter)*len(data_rev):]
			rain_holder_right[-1] = np.mean(rain_holder_right_upper[~np.isnan(rain_holder_right_upper)])-np.mean(rain_holder_right_lower[~np.isnan(rain_holder_right_lower)])

		#rain_holder_id = zip(id,rain_holder)
		
		searcher_left = 0
		searcher_right = 0
		for i in xrange(rain_tip_binary_counter):
			if rain_holder_left[i] > rain_thres: #0.0103: #This is used to remove the doubles in the data, kinda of lag bustin coefficient
				rain_tip_finder_left[id[i]] = 1  #Lets search here again
				searcher_left+=1
			if rain_holder_right[i] > rain_thres: 
				rain_tip_finder_right[id[i]] = 1 
				searcher_right+=1
				
		if output == True: print("Number of searchers:", searcher_left, searcher_right)
		
		if k == len(mul): continue #i.e. exit
		
		#Update for next level of search
		rain_tip_binary_counter*=mul[k]
		rain_tip_finder_guide_left = np.zeros(rain_tip_binary_counter)
		rain_tip_finder_guide_right = np.zeros(rain_tip_binary_counter)
		
		#Determines locations were to search next (i.e. if a previous level search didn't find any significant differences then why bother to continue to search)				

		for i in xrange(len(id)-1):
			if rain_tip_finder_left[i] == 1:
				"""MIGHT HAVE TO CHANGE HERE THE -3, -2, -1 TO A MUTIPLE OF MUL AT INDEX K"""
				
				if search_backward == True:
					for j in xrange(mul[k]):
						rain_tip_finder_guide_left[mul[k]*i-(mul[k]+j-1)] = 1
				if search_forward == True:		
					for j in xrange(2*mul[k]):
						rain_tip_finder_guide_left[mul[k]*i+j] = 1
						
			if rain_tip_finder_right[i] == 1:
				"""MIGHT HAVE TO CHANGE HERE THE -3, -2, -1 TO A MUTIPLE OF MUL AT INDEX K"""
				if search_backward == True:
					for j in xrange(mul[k]):
						rain_tip_finder_guide_right[mul[k]*i-(mul[k]+j-1)] = 1
				if search_forward == True:		
					for j in xrange(2*mul[k]):
						rain_tip_finder_guide_right[mul[k]*i+j] = 1
		
		if rain_tip_finder_left[-1] == 1:
			if search_backward == True:
				for j in xrange(mul[k]):
					rain_tip_finder_guide_left[mul[k]*(len(rain_tip_finder_left)-1)-(mul[k]+j-1)] = 1
			if search_forward == True:		
				for j in xrange(mul[k]):
					rain_tip_finder_guide_left[mul[k]*(len(rain_tip_finder_left)-1)+j] = 1
		
		if rain_tip_finder_right[-1] == 1:
			if search_backward == True:
				for j in xrange(mul[k]):
					rain_tip_finder_guide_right[mul[k]*(len(rain_tip_finder_right)-1)-(mul[k]+j-1)] = 1
			if search_forward == True:		
				for j in xrange(mul[k]):
					rain_tip_finder_guide_right[mul[k]*(len(rain_tip_finder_right)-1)+j] = 1
				
	#Once we have leveled down to 1s resolution then we finish searching
	if output == True: print(rain_tip_binary_counter, len(rain_tip_finder_left), len(rain_tip_finder_right))
	#for i in xrange(raintip_est): #This controls how many values you get out (i.e. the estimated number of tips)
	
	for i in xrange(rain_tip_binary_counter):
		if rain_tip_finder_left[i] == 1:
			#Step_Time.append(0.5*(time[(rain_holder_id[i][0]/rain_tip_binary_counter)*len(data)+1]+time[((rain_holder_id[i][0]+1)/rain_tip_binary_counter)*len(data)]))

			Step_Time_Left = np.append(Step_Time_Left, time[(id[i]/rain_tip_binary_counter)*len(data)+1])
			Rain_Holder_Final_Left = np.append(Rain_Holder_Final_Left, rain_holder_left[i])
			Step_Total_Left+=1
			
			Step_Time_Right = np.append(Step_Time_Right, time[(id[i]/rain_tip_binary_counter)*len(data_rev)+1])
			Rain_Holder_Final_Right = np.append(Rain_Holder_Final_Right, rain_holder_right[i])
			Step_Total_Right+=1
	kill = True					
	
	print(Step_Total_Left, Step_Total_Right)
	print(sorted(np.array(list(set(Step_Time_Left)&set(Step_Time_Right)), dtype=float)))
	sys.exit()
	
	print('No. Tips:', len(rain_tip_finder[rain_tip_finder==1]))
	
	Step_Time = Step_Time[np.argsort(Step_Time)] #Sorts in ascending order the time of each step
	Rain_Holder_Final = Rain_Holder_Final[np.argsort(Step_Time)]
	#print(Step_Time, len(Step_Time))
	#print(len(Step_Time), Rain_Holder_Final)
	#sys.exit()
	return Step_Time, Rain_Holder_Final
	
#def Buffer_Tips_TopDown_2D(data, time=None, output=True, precision=10, rain_thres=0.01, search_backward=False):
def Buffer_Tips_TopDown_2D(time=None, output=True, precision=10, rain_thres=0.01, search_backward=False):
	"""This method provides the best estimate of tip times when the noise of
	the instrument is significant enough to suppress their recognition. When
	the noise of the device is extremely minimal relatively then 
	'Buffer_Tips_Threshold' will probably provide better results. Currently
	this method produces ~95% of tips for moderate amount of noise where the 
	remaining 5% is obscured by false positives which have/have not been 
	removed from end processing.
	
	Parameters
    ----------
    data : numpy array
        1 dimensional data array that represents the data vales of the time 
		series
    time : numpy array, optional
        1 dimensional time series array attached to data stream to determine the timing a step
		was detected
	output : boolean, optional
		Outputs information to the user about the step detection at each iteration

    Returns
    -------
    Step_Time : numpy array
        The index or times of each detected step dependent on whether the 'time' 
		imput is given
	Step_Total : int
		Number of detected steps from the algorithm
	"""

	#Setting up initial variables and values
	#time[0]=time[-1]=0																				# Set the first and last value of the time series equal to 0	
	Step_Total=0   																					# Holds the total number of steps occuring in a day
	rain_tip_binary_counter = int(np.round(data.size/2**precision)) #int(2**np.ceil(np.log2(raintip_est)))	# Inital number of search areas. Exponent determines how many levels
	#rain_tip_binary_counter = int(round(86400/2**7))
	Step_Time = np.array([], dtype=object)																					# Holds the index or time of each step detected)
	rain_tip_finder_guide = np.ones(np.round(data.size/2**precision))										# Used to determine which bin to search. (1) Search; (0) Don't Search. (i.e. initially, since we haven't started searching yet we want to check all bins)
	#rain_tip_finder_guide = np.zeros(int(round(86400/2**7)))
	kill = False																					# Used to stop searching for steps and go to output
	

	
	#Buffer tip times	

	#while raintip_est-Step_Total > 0:
	while kill == False:
		id = range(rain_tip_binary_counter)
		rain_tip_finder = np.zeros(rain_tip_binary_counter)
		rain_holder = np.zeros(rain_tip_binary_counter)
		for i in xrange(rain_tip_binary_counter):
			if rain_tip_finder_guide[i] == 1:
				rain_holder[i] = np.mean(data[((i+1)/rain_tip_binary_counter)*len(data):((i+2)/rain_tip_binary_counter)*len(data)])-np.mean(data[(i/rain_tip_binary_counter)*len(data):((i+1)/rain_tip_binary_counter)*len(data)])
		
		rain_holder_id = zip(id,rain_holder)
		#rain_holder_id = sorted(rain_holder_id,key=lambda l:l[1], reverse=True) #Use this to get best tips out rather than all detected tips
		
		#rain_holder_id = rain_holder_id[:raintip_est]
		
		searcher = 0
		for i in xrange(len(rain_holder_id)):
			if rain_holder_id[i][1] > rain_thres: #0.0103: #This is used to remove the doubles in the data, kinda of lag bustin coefficient
				rain_tip_finder[rain_holder_id[i][0]] = 1  #Lets search here again
				searcher+=1
				
		if output == True: print("Number of searchers:", searcher)
		#Once we have leveled down to 1s resolution then we finish searching
		if rain_tip_binary_counter == int(np.round(len(data)/2**precision)*2**precision):
		#if rain_tip_binary_counter >= 86400:
			if output == True: print(rain_tip_binary_counter)
			#for i in xrange(raintip_est): #This controls how many values you get out (i.e. the estimated number of tips)
			for i in xrange(rain_tip_binary_counter):
				if rain_tip_finder[i] == 1:
					#Step_Time.append(0.5*(time[(rain_holder_id[i][0]/rain_tip_binary_counter)*len(data)+1]+time[((rain_holder_id[i][0]+1)/rain_tip_binary_counter)*len(data)]))
					Step_Time = np.append(Step_Time, time[(rain_holder_id[i][0]/rain_tip_binary_counter)*len(data)+1])
					Step_Total+=1
			kill = True		
		
		#Update for next level of search
		rain_tip_binary_counter*=2
		rain_tip_finder_guide = np.zeros(rain_tip_binary_counter)
		
		#Determines locations were to search next (i.e. if a previous level search didn't find any significant differences then why bother to continue to search)
		for i in xrange(1,len(rain_tip_finder)):
			if rain_tip_finder[i] == 1:
				if search_backward == True:
					rain_tip_finder_guide[2*i-3] = 1
					rain_tip_finder_guide[2*i-2] = 1
					rain_tip_finder_guide[2*i-1] = 1
				rain_tip_finder_guide[2*i] = 1
				rain_tip_finder_guide[2*i+1] = 1
				rain_tip_finder_guide[2*i+2] = 1
				rain_tip_finder_guide[2*i+3] = 1
	
	Step_Time = Step_Time[np.argsort(Step_Time)] #Sorts in ascending order the time of each step
	print(Step_Time)
	
	return Step_Time

def Buffer_Tips_TopDown_2D_Test(data, time=None, output=True, precision=10, rain_thres=0.01, search_backward=False):
	"""This method provides the best estimate of tip times when the noise of
	the instrument is significant enough to suppress their recognition. When
	the noise of the device is extremely minimal relatively then 
	'Buffer_Tips_Threshold' will probably provide better results. Currently
	this method produces ~95% of tips for moderate amount of noise where the 
	remaining 5% is obscured by false positives which have/have not been 
	removed from end processing.
	
	Parameters
    ----------
    data : numpy array
        1 dimensional data array that represents the data vales of the time 
		series
    time : numpy array, optional
        1 dimensional time series array attached to data stream to determine the timing a step
		was detected
	output : boolean, optional
		Outputs information to the user about the step detection at each iteration

    Returns
    -------
    Step_Time : numpy array
        The index or times of each detected step dependent on whether the 'time' 
		imput is given
	Step_Total : int
		Number of detected steps from the algorithm
	"""

	#Setting up initial variables and values

	Step_Total=0   																							# Holds the total number of steps occuring in a day
	rain_tip_binary_counter_x = int(np.round(data.shape[0]/2**precision)) #int(2**np.ceil(np.log2(raintip_est)))	# Inital number of search areas. Exponent determines how many levels
	rain_tip_binary_counter_y = int(np.round(data.shape[1]/2**precision))
	Step_Time = np.array([], dtype=object)																	# Holds the index or time of each step detected)
	Rain_Holder_Final = np.array([], dtype = float)
	rain_tip_finder_guide = np.ones([np.round(data.shape[0]/2**precision), np.round(data.shape[1]/2**precision), 2])										# Used to determine which bin to search. (1) Search; (0) Don't Search. (i.e. initially, since we haven't started searching yet we want to check all bins)
	kill = False
	
	print("rain_tip_binary_counter_x", rain_tip_binary_counter_x)
	print("rain_tip_binary_counter_y", rain_tip_binary_counter_y)	
	it = 0
	while kill == False:
		id = range((rain_tip_binary_counter_x)*(rain_tip_binary_counter_y)*2)
		rain_tip_finder = np.zeros([rain_tip_binary_counter_x, rain_tip_binary_counter_y, 2])
		rain_holder = np.zeros([rain_tip_binary_counter_x, rain_tip_binary_counter_y, 2])
		for i in xrange(rain_tip_binary_counter_x):
			for j in xrange(rain_tip_binary_counter_y):
				if rain_tip_finder_guide[i,j,0] == 1:
					#print("X:", ((i/rain_tip_binary_counter_x)*data.shape[0], ((i+1)/rain_tip_binary_counter_x)*data.shape[0]), " Y:", ((j/rain_tip_binary_counter_y)*data.shape[1], ((j+1)/rain_tip_binary_counter_y)*data.shape[1]), "to X:", (((i+1)/rain_tip_binary_counter_x)*data.shape[0], ((i+2)/rain_tip_binary_counter_x)*data.shape[0]), "Y:", ((j/rain_tip_binary_counter_y)*data.shape[1], ((j+1)/rain_tip_binary_counter_y)*data.shape[1]))
						#print("X:", ((i/rain_tip_binary_counter_x)*data.shape[0], ((i+1)/rain_tip_binary_counter_x)*data.shape[0]), " Y:", ((j/rain_tip_binary_counter_y)*data.shape[1], ((j+1)/rain_tip_binary_counter_y)*data.shape[1]), "to X:", ((i/rain_tip_binary_counter_x)*data.shape[0], ((i+1)/rain_tip_binary_counter_x)*data.shape[0]), "Y:", (((j+1)/rain_tip_binary_counter_y)*data.shape[1], ((j+2)/rain_tip_binary_counter_y)*data.shape[1]))
					
					
					#print(((i+1)/rain_tip_binary_counter_x)*data.shape[0], "-", ((i+2)/rain_tip_binary_counter_x)*data.shape[0], (j/rain_tip_binary_counter_y)*data.shape[1], "-", ((j+1)/rain_tip_binary_counter_y)*data.shape[1])
					#print((i/rain_tip_binary_counter_x)*data.shape[0], "-", ((i+1)/rain_tip_binary_counter_x)*data.shape[0], (j/rain_tip_binary_counter_y)*data.shape[1], "-", ((j+1)/rain_tip_binary_counter_y)*data.shape[1])
					
					
					rain_holder[i,j,0] = np.mean(data[((i+1)/rain_tip_binary_counter_x)*data.shape[0]:((i+2)/rain_tip_binary_counter_x)*data.shape[0], (j/rain_tip_binary_counter_y)*data.shape[1]:((j+1)/rain_tip_binary_counter_y)*data.shape[1]]) - np.mean(data[(i/rain_tip_binary_counter_x)*data.shape[0]:((i+1)/rain_tip_binary_counter_x)*data.shape[0], (j/rain_tip_binary_counter_y)*data.shape[1]:((j+1)/rain_tip_binary_counter_y)*data.shape[1]])
				
				if rain_tip_finder_guide[i,j,1] == 1:	
					rain_holder[i,j,1] = np.mean(data[(i/rain_tip_binary_counter_x)*data.shape[0]:((i+1)/rain_tip_binary_counter_x)*data.shape[0], ((j+1)/rain_tip_binary_counter_y)*data.shape[1]:((j+2)/rain_tip_binary_counter_y)*data.shape[1]]) - np.mean(data[(i/rain_tip_binary_counter_x)*data.shape[0]:((i+1)/rain_tip_binary_counter_x)*data.shape[0], (j/rain_tip_binary_counter_y)*data.shape[1]:((j+1)/rain_tip_binary_counter_y)*data.shape[1]])
						#rain_holder[i,j,1] = np.mean(data[((i+1)/rain_tip_binary_counter_x)*data.shape[0]:((i+2)/rain_tip_binary_counter_x)*data.shape[0], ((j+1)/rain_tip_binary_counter_y)*data.shape[1]:((j+2)/rain_tip_binary_counter_y)*data.shape[1]]) - np.mean(data[(i/rain_tip_binary_counter_x)*data.shape[0]:((i+1)/rain_tip_binary_counter_x)*data.shape[0], (j/rain_tip_binary_counter_y)*data.shape[1]:((j+1)/rain_tip_binary_counter_y)*data.shape[1]])

		#print(rain_tip_finder.shape, rain_holder.shape)
		#rain_tip_finder[np.abs(rain_holder) >= 0.01] = 1
		
		#print(rain_holder.size, len(id))
		rain_holder_id = zip(id,rain_holder)

		
		it += 1
		#if it >= 3: sys.exit()
		print("HERE", rain_tip_binary_counter_x-1, rain_holder.shape)
		
		#print(np.abs(rain_holder))
		#print(rain_tip_finder_guide)
		print(it)
		
		searcher = 0
		for i in xrange(rain_tip_binary_counter_x):
			for j in xrange(rain_tip_binary_counter_y):
				for k in xrange(2):
					
					if np.abs(rain_holder[i,j,k]) > rain_thres: #0.0103: #This is used to remove the doubles in the data, kinda of lag bustin coefficient
						rain_tip_finder[i,j,k] = 1  #Lets search here again
						searcher+=1

		if output == True: print("Number of searchers:", searcher)
		#Once we have leveled down to 1s resolution then we finish searching
		if rain_tip_binary_counter_x == int(np.round(data.shape[0]/2**precision)*2**precision) or rain_tip_binary_counter_y == int(np.round(data.shape[1]/2**precision)*2**precision):
		#if rain_tip_binary_counter >= 86400:
			if output == True: print(rain_tip_binary_counter_x, rain_tip_binary_counter_y)
			#for i in xrange(raintip_est): #This controls how many values you get out (i.e. the estimated number of tips)
			
			for i in xrange(rain_tip_binary_counter_x-1):
				for j in xrange(rain_tip_binary_counter_y-1):
					if rain_tip_finder[i,j,0] == 1:
						#Step_Time.append(0.5*(time[(rain_holder_id[i][0]/rain_tip_binary_counter)*len(data)+1]+time[((rain_holder_id[i][0]+1)/rain_tip_binary_counter)*len(data)]))
						
						Step_Time = np.append(Step_Time, time[(rain_tip_finder[i,j,0]/rain_tip_binary_counter_x)*data.shape[0]+1])
						Rain_Holder_Final = np.append(Rain_Holder_Final, rain_holder[i,j,0])
						Step_Total+=1
					if rain_tip_finder[i,j,1] == 1:
						#Step_Time.append(0.5*(time[(rain_holder_id[i][0]/rain_tip_binary_counter)*len(data)+1]+time[((rain_holder_id[i][0]+1)/rain_tip_binary_counter)*len(data)]))
						Step_Time = np.append(Step_Time, time[(rain_tip_finder[i,j,1]/rain_tip_binary_counter_x)*data.shape[1]+1])
						Rain_Holder_Final = np.append(Rain_Holder_Final, rain_holder[i,j,1])
						Step_Total+=1
						
			kill = True	
			
		#Update for next level of search
		rain_tip_binary_counter_x*=2
		#rain_tip_binary_counter_x-=1
		rain_tip_binary_counter_y*=2
		#rain_tip_binary_counter_y-=1
		rain_tip_finder_guide = np.zeros([rain_tip_binary_counter_x, rain_tip_binary_counter_y, 2])
		
		print("HHHHH", rain_tip_finder_guide.shape, rain_tip_finder.shape) 
		
		#Determines locations were to search next (i.e. if a previous level search didn't find any significant differences then why bother to continue to search)
		for i in xrange(rain_tip_finder.shape[0]):
			for j in xrange(rain_tip_finder.shape[1]):
				for k in xrange(2): #Number of degrees of freedom
					if rain_tip_finder[i,j,k] == 1:
						for l in xrange(2):
							for m in xrange(2):
								rain_tip_finder_guide[2*i+l,2*j+m,k] = 1
		
		# for i in xrange(rain_tip_finder.shape[0]):
			# for j in xrange(rain_tip_finder.shape[1]):
				# if rain_tip_finder[i,j,0] == 1:
					# if search_backward == True:
						# rain_tip_finder_guide[2*i, 2*j-3, 0] = 1
						# rain_tip_finder_guide[2*i, 2*j-2, 0] = 1
						# rain_tip_finder_guide[2*i, 2*j-1, 0] = 1
					# rain_tip_finder_guide[2*i, 2*j, 0] = 1
					# rain_tip_finder_guide[2*i, 2*j+1, 0] = 1
					# rain_tip_finder_guide[2*i, 2*j+2, 0] = 1
					# rain_tip_finder_guide[2*i, 2*j+3, 0] = 1
					
					
				# if rain_tip_finder[i,j,1] == 1:
					# if search_backward == True:
						# rain_tip_finder_guide[2*i-3, 2*j, 1] = 1
						# rain_tip_finder_guide[2*i-2, 2*j, 1] = 1
						# rain_tip_finder_guide[2*i-1, 2*j, 1] = 1
					# #print("2*i+3", 2*i+3)
					# rain_tip_finder_guide[2*i, 2*j, 1] = 1
					# rain_tip_finder_guide[2*i+1, 2*j, 1] = 1
					# rain_tip_finder_guide[2*i+2, 2*j, 1] = 1
					# rain_tip_finder_guide[2*i+3, 2*j, 1] = 1
				
					
		
		#print(rain_tip_finder)		
		#print("AAAAA", rain_tip_finder_guide)
		
		#sys.exit()

	#print(rain_tip_finder, rain_holder)
	#print(rain_tip_finder.shape, rain_holder.shape)
	rain_holder[rain_tip_finder != 1] = 0 
	
	#Used to remove sections of grid underlap REQUIRED FOR NIMROD DATA
	# for i in xrange(it):
		# rain_holder = np.delete(rain_holder, (rain_holder.shape[0]/it)*i, axis=0)
		# rain_holder = np.delete(rain_holder, (rain_holder.shape[0]/it)*i, axis=1)
			
	print("#################")
	print(rain_holder.shape, rain_tip_finder.shape)

	return Step_Time, rain_holder

def Buffer_Tips_TopDown_2D_Test_v2(data, axis_x=None, axis_y=None, output=True, precision=10, rain_thres=0.01, search_backward=False):
	"""This method provides the best estimate of tip times when the noise of
	the instrument is significant enough to suppress their recognition. When
	the noise of the device is extremely minimal relatively then 
	'Buffer_Tips_Threshold' will probably provide better results. Currently
	this method produces ~95% of tips for moderate amount of noise where the 
	remaining 5% is obscured by false positives which have/have not been 
	removed from end processing.
	
	Parameters
    ----------
    data : numpy array
        2 dimensional data array that represents the data vales of the time 
		series
    axis : numpy array, optional
        2 dimensional data array of the boundary values (e.g. Height and Time)
		Must be configured at a Nx2 matrix
	output : boolean, optional
		Outputs information to the user about the step detection at each iteration

    Returns
    -------
    Step_Time : numpy array
        The index or times of each detected step dependent on whether the 'time' 
		imput is given
	Step_Total : int
		Number of detected steps from the algorithm
	"""
	
	#shape[0] = y axis = Height
	#shape[1] = x axis = Time
	# print(data.shape)
	# print(data[100,:].shape)
	# print(data[:,100].shape)
	# sys.exit()
	#Setting up initial variables and values

	Step_Total=0   																							# Holds the total number of steps occuring in a day
	rain_tip_binary_counter_x = int(np.round(data.shape[1]/2**precision)) #int(2**np.ceil(np.log2(raintip_est)))	# Inital number of search areas. Exponent determines how many levels
	rain_tip_binary_counter_y = int(np.round(data.shape[0]/2**precision))
	Step_Time = np.array([], dtype=object)																	# Holds the index or time of each step detected)
	Step_Height = np.array([], dtype=object)	
	Rain_Holder_Final = np.array([], dtype = float)
	#rain_tip_finder_guide = np.ones([np.round(data.shape[1]/2**precision), np.round(data.shape[0]/2**precision)-1, 2])										# Used to determine which bin to search. (1) Search; (0) Don't Search. (i.e. initially, since we haven't started searching yet we want to check all bins)
	rain_tip_finder_guide = np.ones([rain_tip_binary_counter_x,rain_tip_binary_counter_y-1,4])
	kill = False
	
	print("rain_tip_binary_counter_x", rain_tip_binary_counter_x)
	print("rain_tip_binary_counter_y", rain_tip_binary_counter_y)	
	it = 0
	while kill == False:
		id_x = range(rain_tip_binary_counter_x)
		id_y = range(rain_tip_binary_counter_y)
		rain_tip_finder = np.zeros([rain_tip_binary_counter_x, rain_tip_binary_counter_y, 4])
		#rain_holder = np.zeros([rain_tip_binary_counter_x, rain_tip_binary_counter_y, 2])
		rain_holder = np.zeros([rain_tip_binary_counter_x, rain_tip_binary_counter_y-1, 4])
		for i in xrange(rain_tip_binary_counter_x):
			for j in xrange(rain_tip_binary_counter_y-1):
				# print(data[((j+1)/rain_tip_binary_counter_x)*data.shape[1]:((j+2)/rain_tip_binary_counter_x)*data.shape[1], (i/rain_tip_binary_counter_y)*data.shape[0]:((i+1)/rain_tip_binary_counter_y)*data.shape[0]])
				# print(data[(j/rain_tip_binary_counter_x)*data.shape[1]:((j+1)/rain_tip_binary_counter_x)*data.shape[1], (i/rain_tip_binary_counter_y)*data.shape[0]:((i+1)/rain_tip_binary_counter_y)*data.shape[0]])
								
				# print(data[(i/rain_tip_binary_counter_x)*data.shape[1]:((i+1)/rain_tip_binary_counter_x)*data.shape[1], (j/rain_tip_binary_counter_y)*data.shape[0]:((j+1)/rain_tip_binary_counter_y)*data.shape[0]])				
				# print(data[(i/rain_tip_binary_counter_x)*data.shape[1]:((i+1)/rain_tip_binary_counter_x)*data.shape[1], ((j+1)/rain_tip_binary_counter_y)*data.shape[0]:((j+2)/rain_tip_binary_counter_y)*data.shape[0]])
				# print('------------')
				#sys.exit()
				if rain_tip_finder_guide[i,j,0] == 1:	#Looks Up-Down matrix
					rain_holder[i,j,0] = np.mean(data[((j+1)/rain_tip_binary_counter_x)*data.shape[1]:((j+2)/rain_tip_binary_counter_x)*data.shape[1], (i/rain_tip_binary_counter_y)*data.shape[0]:((i+1)/rain_tip_binary_counter_y)*data.shape[0]]) - np.mean(data[(j/rain_tip_binary_counter_x)*data.shape[1]:((j+1)/rain_tip_binary_counter_x)*data.shape[1], (i/rain_tip_binary_counter_y)*data.shape[0]:((i+1)/rain_tip_binary_counter_y)*data.shape[0]])
				
				if rain_tip_finder_guide[i,j,2] == 1:	#Looks Up-Down matrix
					rain_holder[i,j,2] = np.mean(data[(j/rain_tip_binary_counter_x)*data.shape[1]:((j+1)/rain_tip_binary_counter_x)*data.shape[1], (i/rain_tip_binary_counter_y)*data.shape[0]:((i+1)/rain_tip_binary_counter_y)*data.shape[0]]) - np.mean(data[((j+1)/rain_tip_binary_counter_x)*data.shape[1]:((j+2)/rain_tip_binary_counter_x)*data.shape[1], (i/rain_tip_binary_counter_y)*data.shape[0]:((i+1)/rain_tip_binary_counter_y)*data.shape[0]])
				
				if rain_tip_finder_guide[i,j,1] == 1:	#Looks Left-Right matrix
					rain_holder[i,j,1] = np.mean(data[(i/rain_tip_binary_counter_x)*data.shape[1]:((i+1)/rain_tip_binary_counter_x)*data.shape[1], (j/rain_tip_binary_counter_y)*data.shape[0]:((j+1)/rain_tip_binary_counter_y)*data.shape[0]]) - np.mean(data[(i/rain_tip_binary_counter_x)*data.shape[1]:((i+1)/rain_tip_binary_counter_x)*data.shape[1], ((j+1)/rain_tip_binary_counter_y)*data.shape[0]:((j+2)/rain_tip_binary_counter_y)*data.shape[0]])
				
				if rain_tip_finder_guide[i,j,3] == 1:	#Looks Left-Right matrix
					rain_holder[i,j,3] = np.mean(data[(i/rain_tip_binary_counter_x)*data.shape[1]:((i+1)/rain_tip_binary_counter_x)*data.shape[1], ((j+1)/rain_tip_binary_counter_y)*data.shape[0]:((j+2)/rain_tip_binary_counter_y)*data.shape[0]]) - np.mean(data[(i/rain_tip_binary_counter_x)*data.shape[1]:((i+1)/rain_tip_binary_counter_x)*data.shape[1], (j/rain_tip_binary_counter_y)*data.shape[0]:((j+1)/rain_tip_binary_counter_y)*data.shape[0]])
		
		#print(rain_holder)
		
		#print(rain_holder.size, len(id))
		#rain_holder_id = zip(id,rain_holder)
		
		it += 1
		#if it >= 3: sys.exit()
		#print("HERE", rain_tip_binary_counter_x-1, rain_holder.shape)
		
		#print(np.abs(rain_holder))
		#print(rain_tip_finder_guide)
		print(it)
		
		searcher = 0
		for i in xrange(rain_tip_binary_counter_x):
			for j in xrange(rain_tip_binary_counter_y-1):
				for k in xrange(4):
					
					if abs(rain_holder[i,j,k]) > rain_thres: #0.0103: #This is used to remove the doubles in the data, kinda of lag bustin coefficient
						rain_tip_finder[i,j,k] = 1  #Lets search here again
						searcher+=1
		print(rain_tip_finder)
		print(rain_tip_finder.shape)
		

		if output == True: print("Number of searchers:", searcher)
		#Once we have leveled down to 1s resolution then we finish searching
		if rain_tip_binary_counter_x == int(np.round(data.shape[1]/2**precision)*2**precision) or rain_tip_binary_counter_y == int(np.round(data.shape[0]/2**precision)*2**precision):

			if output == True: print(rain_tip_binary_counter_x, rain_tip_binary_counter_y)
			
			for i in xrange(rain_tip_binary_counter_x):
				for j in xrange(rain_tip_binary_counter_y-1):

					if rain_tip_finder[i,j,0] == 1:
						Step_Time = np.append(Step_Time, axis_x[(j/rain_tip_binary_counter_x)*data.shape[1]+1])
						Step_Height = np.append(Step_Height, axis_y[(i/rain_tip_binary_counter_y)*data.shape[0]])
						Rain_Holder_Final = np.append(Rain_Holder_Final, rain_holder[i,j,0])
						Step_Total+=1
					
					if rain_tip_finder[i,j,2] == 1:
						Step_Time = np.append(Step_Time, axis_x[(j/rain_tip_binary_counter_x)*data.shape[1]+1])
						Step_Height = np.append(Step_Height, axis_y[(i/rain_tip_binary_counter_y)*data.shape[0]])
						Rain_Holder_Final = np.append(Rain_Holder_Final, rain_holder[i,j,2])
						Step_Total+=1
					
					if rain_tip_finder[i,j,1] == 1:					
						Step_Time = np.append(Step_Time, axis_x[(i/rain_tip_binary_counter_x)*data.shape[1]])
						Step_Height = np.append(Step_Height, axis_y[(j/rain_tip_binary_counter_y)*data.shape[0]])
						Rain_Holder_Final = np.append(Rain_Holder_Final, rain_holder[i,j,1])
						Step_Total+=1
						
					if rain_tip_finder[i,j,3] == 1:
						Step_Time = np.append(Step_Time, axis_x[(i/rain_tip_binary_counter_x)*data.shape[1]])
						Step_Height = np.append(Step_Height, axis_y[(j/rain_tip_binary_counter_y)*data.shape[0]])
						Rain_Holder_Final = np.append(Rain_Holder_Final, rain_holder[i,j,3])
						Step_Total+=1
						
			kill = True	
			
		#Update for next level of search
		rain_tip_binary_counter_x*=2
		#rain_tip_binary_counter_x-=1
		rain_tip_binary_counter_y*=2
		#rain_tip_binary_counter_y-=1
		rain_tip_finder_guide = np.zeros([rain_tip_binary_counter_x, rain_tip_binary_counter_y-1, 4])
		
		#print("HHHHH", rain_tip_finder_guide.shape, rain_tip_finder.shape) 
		
		#Determines locations were to search next (i.e. if a previous level search didn't find any significant differences then why bother to continue to search)
		for i in xrange(rain_tip_finder.shape[0]):
			for j in xrange(rain_tip_finder.shape[1]):
				for k in xrange(4): #Number of degrees of freedom
					if rain_tip_finder[i,j,k] == 1:
						for l in xrange(5):	# Number of places from centre i, j coords you search (i.e. needs to be expanded as we search deeper)
							for m in xrange(5):
								rain_tip_finder_guide[2*i+l,2*j+m,k] = 1
		

	#print(rain_tip_finder, rain_holder)
	#print(rain_tip_finder.shape, rain_holder.shape)
	
	
	#rain_holder[rain_tip_finder != 1] = 0 
	
	#Used to remove sections of grid underlap REQUIRED FOR NIMROD DATA
	# for i in xrange(it):
		# rain_holder = np.delete(rain_holder, (rain_holder.shape[0]/it)*i, axis=0)
		# rain_holder = np.delete(rain_holder, (rain_holder.shape[0]/it)*i, axis=1)
			
	print("#################")
	print(rain_holder.shape, rain_tip_finder.shape)
	print("Step_Time", Step_Time, len(Step_Time))
	print("Step_Height", Step_Height, len(Step_Height))

	return Step_Time, Step_Height, rain_holder	
############################################################################
"""Algorithm 2: Thresholder"""	
	
def Buffer_Tips_Threshold(rain, time, cal_no, cal_Tipmultipler):
	"""This method uses a threshold value dependant on the calibration 
	multipler which is set slightly lower than its reported tip value to 
	try and account for the inability to add the correct amount of voltage"""
	
	time[0]=time[-1]=0			# Set the first and last value of the time series equal to 0	
	raintot=0   				# Number of tip buckets has occurred in a day

	#Buffer tip times
	for i in range(len(time)-1):
		#if rain[i+1]-rain[i]<0.03:
		if rain[i+1]-rain[i]<(0.15*cal_Tipmultipler[cal_no]): #0.03
			time[i+1]=0
		else:
			raintot+=1

	#Remove traces when tips has not occured
	Tip_Time = time.copy()[time.copy()!=0]

	return Tip_Time, raintot
	
############################################################################
"""Algorithm 3/4: Mean Gradient & Converger"""
	
def Buffer_Tips_MeanGradient(rain, time, cal_no, cal_Tipmultipler, rain_offseter, step_size, standalone = True):
	"""This method is used in conjunction with Buffer_Tip_Converger which
	attempts to alter the tip threshold by small increments to match as 
	closely as possible with the expected number of times calculated by
	the maximum voltage at the end of the day"""
	
	raintot=0
	time[0]=time[-1]=0
	Tip_Time = []
	
	if step_size == 1:
		for i in xrange(1,len(time)-1, step_size): #step size is 2 as our spatial resolution is now 2s rather than 1s
			#if rain[i+1]-rain[i]<((0.1+rain_offseter)*cal_Tipmultipler[cal_no]): #determined from finding transition between normal variations and a tip
			if np.mean(rain[i+1])-np.mean(rain[i])<((0.10+rain_offseter)*cal_Tipmultipler[cal_no]): #!!!NEW!!! method for determines tips more accurately
				time[i+1] = 0
			else:
				raintot+=1
				Tip_Time.append(time[i+1])
			
	elif step_size == 2:
		for i in xrange(1,len(time)-1, step_size): #step size is 2 as our spatial resolution is now 2s rather than 1s
			#if rain[i+1]-rain[i]<((0.1+rain_offseter)*cal_Tipmultipler[cal_no]): #determined from finding transition between normal variations and a tip
			if np.mean([rain[i], rain[i+1]])-np.mean([rain[i],rain[i-1]])<((0.10+rain_offseter)*cal_Tipmultipler[cal_no]): #!!!NEW!!! method for determines tips more accurately
				time[i+1] = 0
				time[i+2] = 0
			else:
				raintot+=1
				Tip_Time.append(time[i+1])
	
	if standalone == True:
		return Tip_Time, raintot
	else:
		return time, raintot
		
def Buffer_Tips_Converger(rain, time, cal_no, cal_Tipmultipler, step_size):

	######################################################################
	"""Determine the rain rate from an analogy input"""
	
	time[0]=time[-1]=0									# Set the first and last value of the time series equal to 0	
	raintot=0   										# Number of tip buckets has occurred in a day
	raintip_est = Rain_Tip_Estimator(rain, cal_no, cal_Tipmultipler)
	norain = False										# used to indicate if any rain was found
	rain_offseter = 0									# value used to change the threshold for determining tips
	rain_tip_trys = 0									# Number of tries to find tips
	buffer_tips_force = False							# Used to force system out of buffer tip loop if it can't find sutiable number of tips
	first_run = True
	rain_tip_coeff = []
	rain_tip_diff = []
	
	#Buffer tip times
	print(np.max(rain))
	print(cal_Tipmultipler[cal_no])
	print("Rain Tip Estimator:", raintip_est)
	
	#First Run (Even if no tips are found its still important for reducing Tip_Time to 0
	time, raintot = Buffer_Tips_MeanGradient(rain, time, cal_no, cal_Tipmultipler, rain_offseter, step_size, standalone = False)
	if raintot == 0: norain = True
	rain_tip_coeff.append(rain_offseter)
	rain_tip_diff.append(np.abs(raintip_est-raintot))
	rain_tip_trys += 1

	#Loop'd run to determine a sensible number of tips in relation to the final voltage. i.e. its a Newtonian style converger
	while (rain_tip_diff[-1] > 10 and norain == False and buffer_tips_force == False):
		time, raintot = Buffer_Tips_MeanGradient(rain, time, cal_no, cal_Tipmultipler, rain_offseter, step_size, standalone = False)	
		rain_tip_coeff.append(rain_offseter)
		rain_tip_diff.append(np.abs(raintip_est-raintot))
		rain_tip_trys += 1
		print("raintip_est-raintot", np.abs(raintip_est-raintot))
		
		#Determines if there was rain and if not whether there are large differences from expected
		if raintot == 0:
			norain = True
			print(norain)
		elif raintip_est-raintot < -10:
			rain_offseter += 0.01
		elif raintip_est-raintot > 10:
			rain_offseter -= 0.01
		if rain_tip_trys > 15:
			rain_offseter = rain_tip_coeff[rain_tip_diff.index(min(rain_tip_diff))]
			time, raintot = Buffer_Tips_MeanGradient(rain, time, cal_no, cal_Tipmultipler, rain_offseter, step_size, standalone = False)
			buffer_tips_force = True
	
	Tip_Time = time.copy()[time.copy()!=0]
	
	return Tip_Time, raintot

############################################################################
"""Algorithm 5: Sliding window"""
	
def Buffer_Tips_MovingAverage(rain, time, cal_no, cal_Tipmultipler, MA):
	"""This method uses a rolling average to test the statistical difference
	between each bin. A usual 95% confidence level is used. MA is the size 
	of the rolling average"""
	from scipy import stats
	
	Tip_Time = np.zeros(len(rain)-MA)
	raintot = 0
	ttest = np.zeros(len(rain)-MA)
	time_ttest = np.zeros(len(rain)-MA)
	for i in xrange(len(rain)-MA):
		if stats.ttest_ind(rain[i+1:i+1+MA], rain[i:i+MA], equal_var = True)[1] < 0.38: 
			Tip_Time[i] = time[i+1]
			raintot += 1
			
	Tip_Time = Tip_Time[Tip_Time>0]
		# ttest[i] = stats.ttest_ind(rain[i+1:i+1+MA], rain[i:i+MA])[1]
		# #time_ttest[i] = time[i+0.5*MA]
		# time_ttest[i] = time[i+1]
		# if stats.ttest_ind(rain[i+1:i+1+MA], rain[i:i+MA], equal_var = True)[1] < 0.2: 
			# #Tip_Time.append(time[i+MA/2])
			# time_ttest[i]
			# raintot += 1
	
	# raintip_est = Rain_Tip_Estimator(rain, cal_no, cal_Tipmultipler)
	
	# ttest=ttest[~np.isnan(ttest)]
	# time_ttest = time_ttest[~np.isnan(ttest)]
	
	# ttestandtime = np.asarray(zip(time_ttest,ttest))
	# ttestandtimesort = ttestandtime[np.argsort(ttestandtime[:, 1])]
	
	# for i in xrange(len(ttestandtimesort)):
		# if ttestandtimesort[i, 1] <0.38:
			# Tip_Time.append(ttestandtimesort[i, 0])
			# raintot += 1
	
	Tip_Time.sort()
	
	#n, bins, patches = plt.hist(ttest,100)
	
	#plt.xlabel('Smarts')
	#plt.ylabel('Probability')
	#plt.title('Histogram of TTest')
	#plt.axis([0, 1, 0, 100])
	#plt.grid(True)
	#plt.show()
			
	return Tip_Time, raintot

############################################################################
"""Algorithm 6: Bottom-Up"""
	
def Buffer_Tips_BottomUp(rain, time, cal_no, cal_Tipmultipler):
	"""This method provides another estimate for tips using a complex algorithm...
	well at least more complex than a simple threshold anyway. This method involves,
	assuming every single data point has a step in it. Testing everypoint first we 
	slowly move outwards removing data points that are not controlled by a tip"""
	
	time[0]=time[-1]=0																# Set the first and last value of the time series equal to 0	
	raintot=0   																	# Number of tip buckets has occurred in a day
	rain_tip_binary_counter = len(rain)-1
	rain_tip_finder_guide = np.zeros(len(rain)-1)
	rain_tip_finder_guide[:] = 1
	initial = True																	# States the first run
	kill = False
	cyc = 0																			# States number of iterations we've made. Needed for time_holder_initial
	
	#Buffer tip times
	raintip_est = Rain_Tip_Estimator(rain, cal_no, cal_Tipmultipler)
	print("Rain Tip Estimator:", raintip_est)
	
	while kill == False:
		rain_tip_finder = np.zeros(rain_tip_binary_counter)
		time_holder = np.zeros(rain_tip_binary_counter)
		for i in xrange(rain_tip_binary_counter):
			if ((np.mean(rain[((i+1)/rain_tip_binary_counter)*len(rain):((i+2)/rain_tip_binary_counter)*len(rain)])-np.mean(rain[(i/rain_tip_binary_counter)*len(rain):((i+1)/rain_tip_binary_counter)*len(rain)]))>0) and (rain_tip_finder_guide[i] == 1): #0.03
				rain_tip_finder[i]         			= 1
				time_holder[i]             			= time[i]
			elif initial == False:
				for j in xrange(int(2**cyc)):
					time_holder_initial[i*2**cyc+j] = 0 
		
		if initial == True:
			time_holder_initial = time_holder
			initial = False
		
		if rain_tip_binary_counter <= 9:
			print(rain_tip_binary_counter)
			Tip_Time = time_holder_initial[time_holder_initial>0]
			raintot = len(Tip_Time)
			kill = True	
		
		#Update for next level of search
		cyc += 1
		rain_tip_binary_counter = int(np.round(rain_tip_binary_counter/2))
		rain_tip_finder_guide = np.zeros(rain_tip_binary_counter)
		
		#Determines locations were to search next (i.e. if a previous level search didn't find any significant differences then why bother to continue to search)
		for i in xrange(len(rain_tip_finder)-2):
			if rain_tip_finder[i] == 1:
				rain_tip_finder_guide[int(np.floor(i/2))] = 1
				rain_tip_finder_guide[int(np.floor((i+1)/2))] = 1
	
	return Tip_Time, raintot


	
############################################################################
"""Algorithm 7: Mallat-Zhong

Uses a multiscale product smoother with edge preserver and then using a simple
threshold limit a step can be found. Also the polarity and magnitude can also
be extracted from the data if needs be
"""
############################################################################

def Buffer_Tips_Mallat(rain, time):
	rain_smooth = mz_fwt(rain)
	Tip_Time_Temp = find_steps(rain_smooth, 0.001)
	Tip_Time = np.zeros(len(Tip_Time_Temp))
	for i in xrange(len(Tip_Time_Temp)):
		Tip_Time[i] = time[Tip_Time_Temp[i]+3]
	raintot = len(Tip_Time)
	
	return Tip_Time, raintot
	
def mz_fwt(x, n=2):
    """
    Computes the multiscale product of the Mallat-Zhong discrete forward
    wavelet transform up to and including scale n for the input data x.
    If n is even, the spikes in the signal will be positive. If n is odd
    the spikes will match the polarity of the step (positive for steps
    up, negative for steps down).

    This function is essentially a direct translation of the MATLAB code
    provided by Sadler and Swami in section A.4 of the following:
    http://www.dtic.mil/dtic/tr/fulltext/u2/a351960.pdf

    Parameters
    ----------
    x : numpy array
        1 dimensional array that represents time series of data points
    n : int
        Highest scale to multiply to


    Returns
    -------
    prod : numpy array
        The multiscale product for x

    """
    N_pnts   = x.size
    lambda_j = [1.5, 1.12, 1.03, 1.01][0:n]
    if n > 4:
        lambda_j += [1.0]*(n-4)
    
    H = np.array([0.125, 0.375, 0.375, 0.125])
    G = np.array([2.0, -2.0])
    
    Gn = [2]
    Hn = [3]
    for j in range(1,n):
        q = 2**(j-1)
        Gn.append(q+1)
        Hn.append(3*q+1)

    S    = np.concatenate((x[::-1], x))
    S    = np.concatenate((S, x[::-1]))
    prod = np.ones(N_pnts)
    for j in range(n):
        n_zeros = 2**j - 1
        Gz      = _insert_zeros(G, n_zeros)
        Hz      = _insert_zeros(H, n_zeros)
        current = (1.0/lambda_j[j])*np.convolve(S,Gz)
        current = current[N_pnts+Gn[j]:2*N_pnts+Gn[j]]
        prod    *= current
        if j == n-1:
            break
        S_new   = np.convolve(S, Hz)
        S_new   = S_new[N_pnts+Hn[j]:2*N_pnts+Hn[j]]
        S       = np.concatenate((S_new[::-1], S_new))
        S       = np.concatenate((S, S_new[::-1]))
    return prod


def _insert_zeros(x, n):
    """
    Helper function for mz_fwt. Splits input array and adds n zeros
    between values.
    """
    newlen       = (n+1)*x.size
    out          = np.zeros(newlen)
    indices      = range(0, newlen-n, n+1)
    out[indices] = x
    return out


def find_steps(array, threshold):
    """
    Finds local maxima by segmenting array based on positions at which
    the threshold value is crossed. Note that this thresholding is 
    applied after the absolute value of the array is taken. Thus,
    the distinction between upward and downward steps is lost. However,
    get_step_sizes can be used to determine directionality after the
    fact.

    Parameters
    ----------
    array : numpy array
        1 dimensional array that represents time series of data points
    threshold : int / float
        Threshold value that defines a step


    Returns
    -------
    steps : list
        List of indices of the detected steps

    """
    steps        = []
    array        = np.abs(array)
    above_points = np.where(array > threshold, 1, 0)
    ap_dif       = np.diff(above_points)
    cross_ups    = np.where(ap_dif == 1)[0]
    cross_dns    = np.where(ap_dif == -1)[0]
    for upi, dni in zip(cross_ups,cross_dns):
        steps.append(np.argmax(array[upi:dni]) + upi)
    return steps


def get_step_sizes(array, indices, window=1000):
    """
    Calculates step size for each index within the supplied list. Step
    size is determined by averaging over a range of points (specified
    by the window parameter) before and after the index of step
    occurrence. The directionality of the step is reflected by the sign
    of the step size (i.e. a positive value indicates an upward step,
    and a negative value indicates a downward step). The combined 
    standard deviation of both measurements (as a measure of uncertainty
    in step calculation) is also provided.

    Parameters
    ----------
    array : numpy array
        1 dimensional array that represents time series of data points
    indices : list
        List of indices of the detected steps (as provided by 
        find_steps, for example)
    window : int, optional
        Number of points to average over to determine baseline levels
        before and after step.


    Returns
    -------
    step_sizes : list
        List of the calculated sizes of each step
    step_error : list

    """
    step_sizes = []
    step_error = []
    indices    = sorted(indices)
    last       = len(indices) - 1
    for i, index in enumerate(indices):
        if i == 0:
            q = min(window, indices[i+1]-index)
        elif i == last:
            q = min(window, index - indices[i-1])
        else:
            q = min(window, index-indices[i-1], indices[i+1]-index)
        a = array[index:index+q]
        b = array[index-q:index]
        step_sizes.append(a.mean() - b.mean())
        step_error.append(sqrt(a.var()+b.var()))
    return step_sizes, step_error

def get_multipler(array, m=2):
	"""Determines mutipler at each step which produces a whole
	number.
	
	Parameters
	----------
	array : int
		The number you want to divide by
	m : int, optional
		Initial division number. By default this is set to 2
		as this is the lowest integer divisor
		
	Returns
	-------
	mul : numpy array
		The array sequence of multiplers at each step. This will be
		in reversed order so if we start at the lowest number (i.e.
		first number in mul) then we can mutply by the first number 
		in mul and so on
	
	"""
	
	#if number is prime then remove 1 from sequence
	if is_prime(array) == True: array += 1
	
	mul = np.array([], dtype=int)
	while array > m:
		array_temp = array/m
		if array_temp == int(array_temp):
			array//=m
			mul = np.append(mul, m)
		else:
			while array_temp != int(array_temp):
				m += 1
				array_temp = array/m
			array//=m
			mul = np.append(mul, m)

	mul = np.array(list(reversed(mul)), dtype=int)
	
	return mul

def is_prime(a):
    return all(a % i for i in xrange(2, a))
	
if __name__ == "__main__":
	data = np.ones([4,4])
	it=0
	for i in xrange(data.shape[0]):
		for j in xrange(data.shape[1]):
			it+=1
			data[i,j] = it
	
	
	#data = np.array([[0,0,0,0],[0,10,0,0],[0,0,0,0],[0,0,0,0]])
	
	val_x = 1000
	val_y = 500
	data = np.zeros([val_x,val_y])
	data[np.random.randint(val_x), np.random.randint(val_y)] = 1000
	
	# print(data[1,1], data[2,2])

	Step_Time, Step_Height, rain_holder = Buffer_Tips_TopDown_2D_Test_v2(data, np.arange(0,val_x), np.arange(0,val_y), False, 7, 0, True)
	
	#data = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[1,1,1,10,1,1],[0,0,0,0,0,0],[1,1,1,1,1,1],[1,1,1,1,1,1]])
	#print(data)
	#Step_Time, Step_Height, rain_holder = Buffer_Tips_TopDown_2D_Test_v2(data, np.arange(0,data.shape[0]), np.arange(0,data.shape[1]), False, 1, 1, True)
	
	for i in xrange(len(Step_Time)):
		print(data[Step_Time[i],Step_Height[i]])