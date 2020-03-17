# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:59:11 2019

@author: Ollie

On running, code will work through the exercise requiring some user inputs
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score

plt.rcParams.update({'font.size': 16})

def user_input(text,datatype): #function to allow user input for specified text 
    #and required data type
    UserInput = False
    while UserInput != True:    #while loop to make input robust
        try:
            Input = datatype(input(text))
        except ValueError:
            print("Invalid inputs")
        else:
            UserInput = True
            return Input

def function(x, which):
    #function to return a value for an input x depending on which function
    if which == 'sine':
        return np.sin(x)
    elif which == 'test':
        return x
    elif which == 'exp':
        y = []
        for i in range(len(x)):            
            y.append( - x[i] / 0.00055)
        return np.exp(y) / 0.00055

def invert_analytically_sine(lower_bound, upper_bound, quantity):
    #function to invert a quantity of genearted numbers analytically
    #returns an array of random numbers proportional to sin(theta)
    tic = time.time()
    generated_nums = np.random.uniform(lower_bound, upper_bound, quantity)
    inverted_nums = []
    for i in range(quantity):
        inverted_nums.append(np.arccos(1 - 2*generated_nums[i]))
    print("Analytical inversion took:", time.time() - tic, "s")
    return inverted_nums

def test_invert_analytically():
    #function to test the inversion function is working as expected by 
    #generating zeros and checking the inversion is zero
    test = np.average(invert_analytically_sine(0, 0, 1))
    if test == 0:
        return True
    else:
        return False
    
def reject_accept(function_min, function_max, domain_min, domain_max,
                  quantity, which):
    #function to invert quantity of generated numbers by rejection sampling
    #returns an array of random numbers proportional to function f in domain
    #range in function range
    tic = time.time()
    generated_nums1 = np.random.uniform(domain_min, domain_max, 8*quantity)
    generated_nums2 = np.random.uniform(function_min, function_max, 8*quantity)
    accepted_nums = []
    accepted = 0
    rejected = 0
    for i in range(len(generated_nums1)):
        if generated_nums2[i] <= function(generated_nums1[i], which):
            accepted_nums.append(generated_nums1[i])
            accepted += 1
            if accepted >= quantity:
                print("Reject-accept took:", time.time() - tic, "s")
                print("Number of rejections:", rejected)
                return accepted_nums, accepted
        else:
            rejected += 1
    print("Reject-accept took:", time.time() - tic, "s")
    return accepted_nums, accepted

def test_reject_accept():
    #function to test reject-accept function with y=x function where generated
    #number ranges do not overlap
    test = reject_accept(2, 3, 0, 1, 1, 'test')[1]
    if test == 0:
        return True
    else:
        return False

def bin_testing_fit(counts, bins, which):
    #finding error from the bin testing for input bin array and theta values
    #to check it is in distribution of f
    bins_new = []
    for j in range(len(bins) - 1):    #find middle of bins
        bins_new.append((bins[j] + bins[j+1])/2)
    r_squared = r2_score(function(bins_new, which), counts)
    return r_squared 

def nuclei_distance(quantity):
    #returns decay time and distance from detector array for inpout quantity
    mean_time = 0.00055 #s   
    detector_dist = 2 #m
    nuclei_speed = 2000 #m/s
    decay_time = np.random.exponential(mean_time, quantity)
    distance_from_array = []
    for i in range(len(decay_time)):
        distance_from_inject = nuclei_speed * decay_time[i]
        if detector_dist > distance_from_inject:
            distance_from_array.append(detector_dist - distance_from_inject) 
    return distance_from_array, decay_time

def gamma_direction(distance_from_array):
    #function to produce random angles to model gamma emission
    #theta is distributed proportional to sine to avoid clumping at the poles
    theta = invert_analytically_sine(0, 1, len(distance_from_array))
    #phi is distributed between 0 and 2pi uniformly
    phi = np.random.uniform(0, 2 * np.pi, len(distance_from_array))
    return theta, phi
    
def detected_gamma(theta, phi, distance_from_array):
    #function to find the position on the detector where the gamma will land
    x = distance_from_array * np.tan(phi)
    y = distance_from_array / (np.tan(theta) * np.cos(phi))
    z = distance_from_array * np.cos(theta)
    return x, y, z

def smearing(x, y):
    #return new detector position after smearing on making contact
    x_resolution = 0.1 #m
    y_resolution = 0.3 #m
    x_smear = np.random.normal(x, x_resolution)
    y_smear = np.random.normal(y, y_resolution)
    return x_smear, y_smear

def gamma_hit_target(x, y, z, x_length, y_length):
    #function returning the positions of the gamma rays that hit the array
    x_hit = []
    y_hit = []
    hit_target = 0
    missed_target = 0
    for i in range(len(x)):
        if -x_length / 2 <= x[i] <= x_length / 2 and \
            -y_length / 2 <=y[i] <= y_length / 2 and \
                z[i] >= 0:
            x_hit.append(x[i])
            y_hit.append(y[i])
            hit_target += 1
        else:
            missed_target += 1
    return x_hit, y_hit, hit_target, missed_target

def test_gamma_hit():
    #func to test gamma hit function by throwing points that should def miss
    x = np.random.uniform(10, 15, 10)
    y = np.random.uniform(10, 15, 10)
    z = 1
    test = gamma_hit_target(x, y, z, 15, 15)[2]
    if test == 0:
        return True
    else:
        return False

def signal_events(cross_section, quantity):
    #function returning signal events for input luminosity uncertainty, cross 
    #section and quantity of numbers
    lum = 12 #nb
    lum_spread = 0.12
    luminosity = np.random.normal(lum, lum_spread, quantity)
    return np.random.poisson(luminosity * cross_section)

def background_events(quantity):
    #function returning background events for input quantity of numbers
    expected_background = 5.7
    expected_background_var = 0.4
    return np.random.poisson(np.random.normal(expected_background, 
                                              expected_background_var, 
                                              quantity))

def confidence_level(measured_value, events):
    #function to return the confidence value over measured of an input array 
    signal_above_measurement = len(events[np.where(events > measured_value)])
    return (signal_above_measurement / len(events) * 100)

def test_confidence_level():
    #function testing confidence level func by checking confidence of there 
    #being >= 0 events - should be 100 %
    test = confidence_level(-1, signal_events(0.1, 100))
    if test == 100:
        return True
    else:
        return False
            
print("Task 1, generating random distributions proportional to sine:\n")
    
if test_invert_analytically() == True:
    print("Analytical inversion function working")
else:
    print("Analytical inversion function not working\n")   
    
if test_reject_accept() == True:
    print("Reject-accept function working")
else:
    print("Reject-accept function not working\n")
    
quantity = user_input("Input quantity of random numbers to be generated:", int)
num_bins = user_input("Input number of bins for histogram:", int)

#calculating histograms for both distributions
counts, bins, patches = plt.hist(invert_analytically_sine(0, 1, quantity), 
                                         bins = num_bins, alpha = 0.8,
                                         label = 'Analytical', density = True,
                                         color = 'b')
print("Analytical fit R2:",  bin_testing_fit(counts*2,bins,'sine'), "\n")

counts_r, bins_r, patches_r = plt.hist(reject_accept(0, 1, 0, np.pi, 
                                                quantity, 'sine')[0],
                                               bins = num_bins, alpha = 0.8,
                                               label = 'Reject-accept', 
                                               density = True, color = 'r')                                                     
print("Reject-accept fit R2:", bin_testing_fit(counts_r*2, bins_r, 'sine'))            

plt.legend()
x = np.linspace(0,np.pi,quantity)
plt.xlabel("x")
plt.ylabel("sin(x)/2")
plt.plot(x,function(x,'sine')/2)
plt.show()    
    
print("Radioative decay simulation:\n")

print("Testing exponential decay distribution:")
    
counts, bins, patches = plt.hist(nuclei_distance(100000)[1], 
                                 bins = 100, color = 'b', density = True)
plt.xlabel("Time to decay")
plt.ylabel("Number of decays")
x = np.linspace(0, 0.01, 1000)
plt.plot(x, function(x, 'exp'))
plt.xlim(0, 0.001)  #time taken to travel 2m at 2000 m/s
plt.show()

print("Exponential fit R2:", bin_testing_fit(counts, bins, 'exp')) 

if test_gamma_hit() == True:
    print("Detector array function working")
else:
    print("Detector array function not working")

quantity = user_input("Input quantity of random numbers to be generated:", int)
num_bins = user_input("Input number of bins for histograms:", int)
x_length = user_input("Input square array size (m):", float) #m
y_length = x_length #m

distance_from_array = nuclei_distance(quantity)[0]
theta, phi = gamma_direction(distance_from_array)
x, y, z = detected_gamma(theta, phi, distance_from_array)
x_hit, y_hit, hit_target, missed_target = gamma_hit_target(x, y, z, 
                                                           x_length, y_length)
x_smear, y_smear = smearing(x_hit, y_hit)

print("\nNumber of gamma rays that hit target:", hit_target)
print("Number of gamma rays that missed target:", missed_target)
print("Number of nuclei not decayed before array:", 
      quantity - hit_target - missed_target)

#create axis for 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("equal")

#plot points emitted from decay
ax.scatter([np.sin(theta)*np.cos(phi)], 
            [np.sin(theta)*np.sin(phi)], 
            [np.cos(theta)], color = "r", s = 5)

#plot sphere wire frame
phi_sphere, theta_sphere = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
ax.plot_wireframe(np.sin(theta_sphere)*np.cos(phi_sphere),
                  np.sin(theta_sphere)*np.sin(phi_sphere),
                  np.cos(theta_sphere), color = "k", linewidth = 0.5)  
ax.dist = 11
plt.show()

#plotting detector array with 2d histogram
plt.hist2d(x_smear, y_smear, bins = num_bins)
plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.gca().set_aspect('equal', adjustable='box')
cbar = plt.colorbar()
cbar.set_label("Counts")
plt.show()

#plotting 1d histograms to see statistics
counts_x, bins_x, patches_x = plt.hist(x_smear, bins = num_bins, 
        alpha = 0.6, label = 'X', density = True, color = 'b')    
counts_y, bins_y, patches_y = plt.hist(y_smear, bins = num_bins, 
        alpha = 0.6, label = 'Y', density = True, color = 'r')

plt.xlabel("Distance (m)")
plt.ylabel("Normalised counts")
plt.legend()
plt.show()    

print("Modelling statistics of particle physics experiments:\n")
    
if test_confidence_level() == True:
    print("Confidence level function working")
else:
    print("Confidence level function not working")

quantity = user_input("Input quantity of generated experiments:", int)
cross_section = user_input("Input cross section (nb):", float) #nb

signal = signal_events(cross_section, quantity)
background = background_events(quantity)

#using histograms for statistics
counts, bins, patches = plt.hist(signal, bins = np.amax(signal), alpha = 0.6, 
                                 label = 'signal', density = True, color = 'r')
counts_back, bins_back, patches_back = plt.hist(background, 
                                                bins = np.amax(background), 
                                                alpha = 0.6, 
                                                label = 'background', 
                                                density = True, color = 'b')

total_events = signal + background

counts_tot, bins_tot, patches_tot = plt.hist(total_events, 
                                             bins = np.amax(total_events),
                                             alpha = 0.6, label = 'total',
                                             density = True, color = 'g')
plt.legend()
plt.ylabel("Normalised count")
plt.xlabel("Number of events")
plt.xlim(0,25)

print("Cross-section is below:", cross_section, "nb", 
      "\nwith confidence level:", confidence_level(5, total_events), "%")
plt.show()