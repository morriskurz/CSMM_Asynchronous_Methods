# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:04:31 2018

@author: morris.kurz.adm
"""
# Fragen:
# Norm für Fehler
# normaler time_Step Wert

# TODO: maybe implement as Pool to get return values
# save return values
# plot
# DONT SAVE EVERYTHING
from multiprocessing import Process, Queue, Pipe
import numpy as np
from time import time
import matplotlib.pyplot as plt
import math



def solve(initial_cond, limit, pipeline, max_delay, process_const):
    """
        Maximum delay at time n: n-max_delay.
        returns:
            [
            times: amount of time needed to reach step x
            values: values at step x]
    """
    def fds(points, process_const):
        # Here: simple heat equation with discretization with first-order forward
        # difference in time and second-order zentral difference in space
        left, middle, right = points
        delta_x, delta_t = process_const
        temp = delta_t/delta_x/delta_x
        return middle + temp*(right - 2*middle + left)
    tic = time()
    print(tic)
    # Where the result is saved
    result = []
    # Index of current ghost point front
    index_right = 0
    index_left = 0
    # Extract information from variables
    lower_limit, upper_limit = limit
    time_steps, delta_x, delta_t = process_const
    # Time steps at which we want to measure time passed
    times = list(range(0, time_steps, int((time_steps/100))))[1:]
    times_index = 1
    # Define the (for the pe) relevant grid with 2 ghost points
    grid = np.zeros((time_steps, upper_limit-lower_limit+2))

    grid[0, 1:len(initial_cond)+1] = initial_cond
    grid[0, 0] = pipeline[0].recv()
    grid[0, -1] = pipeline[2].recv()
    for i in range(1, time_steps):
        for j in range(2, len(initial_cond)):
            # For inner points, normal finite difference scheme
            grid[i, j] = fds((grid[i-1, j-1], grid[i-1, j], grid[i-1, j+1]), (delta_x, delta_t))
        # Special handling for delay left; block process if max_delay is 
        # exceeded.
        
        while (pipeline[0].poll() and index_left < i-1) or (i-index_left>max_delay+1):
            grid[index_left+1, 0] = pipeline[0].recv()
            index_left += 1
        grid[i, 1] = fds((grid[index_left, 0], grid[i-1, 1], grid[i-1, 2]), (delta_x, delta_t))
        pipeline[1].send(grid[i, 1])
        while (pipeline[2].poll() and index_right < i-1) or (i-index_right>max_delay+1):
            grid[index_right+1, -1] = pipeline[2].recv()
            index_right += 1
        grid[i, -2] = fds((grid[i-1, -3], grid[i-1, -2], grid[index_right, -1]), (delta_x, delta_t))
        pipeline[3].send(grid[i, -2])
        if i == times[times_index]:
            times_index = min(times_index+1, len(times)-1)
            result.append([(time()-tic), grid[i, 1:len(initial_cond)+1]])
    #print(grid)
    #print(result)
    
def get_initial_cond(grid_resolution):
    return list(map(
            lambda x: 10*math.sin(2*2*math.pi*x/grid_resolution + 0.5),
            list(range(grid_resolution))))
    
def exact_solution(time):
    temp = []
    t = delta_t*time
    for i in range(grid_resolution):
        x = delta_x*i
        sol = math.exp(-alpha*(wavenumber**2)*t)*amplitude*math.sin(
                wavenumber*x+phase_angle-c*t)
        temp.append(sol)
    return temp
    
time_steps = 50000
grid_resolution = 256
delta_x = 2*math.pi/grid_resolution
delta_t = 10e-5
amount_pe = 4

if __name__ == '__main__':
    # Get initial conditions
    initial_cond = get_initial_cond(grid_resolution)
    # Ranges of the processing elements
    pe_ranges = []
    # Processing elements
    pe = []
    # Using pipelines for communication, each process need 2 receiving
    # pipelines (ghost points) and 2 outgoing pipelines corresponding to
    # adjacent boundary points.
    # Structure: [Receiving information from the left point, where to write information 
    # for the left point, Information from the right point, where to .. right]
    pipeline = [[None, None, None, None] for _ in range(amount_pe)]
    # TODO: Write testcase for veryfication
    for i in range(amount_pe):
         left_read, left_write = Pipe()
         right_read, right_write = Pipe()
         # Information should go to the left point, where he reads the 
         # information from his right point
         pipeline[i-1][2] = left_read
         # Structure defined above
         pipeline[i][1] = left_write
         # Analogous to above, just modulo amount_pe to not get IndexError
         pipeline[(i+1)%amount_pe][0] = right_read
         pipeline[i][3] = right_write

    # Length of each processing element
    length_pe = grid_resolution//amount_pe
    for i in range(amount_pe):
        if i == amount_pe-1:
            # If on the border of the domain, use rightmost point
            boundary = grid_resolution-1
        else:
            # Otherwise use next point
            boundary = (i+1)*length_pe-1
        # Ghost points for boundary points, here the computated 
        # information gets stored
        pipeline[i][1].send(initial_cond[i*length_pe])
        #print(pipeline[i-1][2].recv()==initial_cond[i*length_pe])
        pipeline[i][3].send(initial_cond[boundary])
        # Save information on pe boundaries
        pe_ranges.append((i*length_pe, boundary))
    tic = time()
    for i in range(amount_pe):
        # Define processes with initial conditions, range, pipelines and delays
        
        pe.append(Process(target=solve,
                          args=(initial_cond[pe_ranges[i][0]:pe_ranges[i][1]+1],
                                pe_ranges[i], pipeline[i], 100000,
                                [time_steps, delta_x, delta_t])))
    [proc.start() for proc in pe]
    [proc.join(10) for proc in pe]
    print(time()-tic)