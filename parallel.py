# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:04:31 2018

@author: morris.kurz.adm
"""
# Fragen:
# Norm für Fehler -> 2, inf
# normaler time_Step Wert

# TODO: maybe implement as Pool to get return values
# save return values
# plot
from multiprocessing import Process, Queue, Pipe, Pool
import numpy as np
from time import time
import matplotlib.pyplot as plt
import math



def solve(process):
    """
        Maximum delay at time n: n-max_delay.
        returns:
            [
            times: amount of time needed to reach step x
            values: values at step x]
    """
    initial_cond, limit, pipeline, max_delay, process_const = process
    def fds(points, process_const):
        # Here: simple heat equation with discretization with first-order forward
        # difference in time and second-order zentral difference in space
        left, middle, right = points
        delta_x, delta_t = process_const
        # For now, it is constant 0.1
        #temp = delta_t/delta_x/delta_x
        return middle + r_alpha*(right - 2*middle + left)
    tic = time()
    print(tic)
    # Where the result is saved
    result = []
    # Index of current ghost point front
    index_right = 0
    index_left = 0
    # Extract information from variables
    lower_limit, upper_limit = limit
    time_steps, delta_x, delta_t, r_alpha = process_const
    # Time steps at which we want to measure time passed
    times = list(range(0, time_steps, int((time_steps/1000))))
    times_index = 1
    result.append([0, initial_cond])
    # Define the (for the pe) relevant grid with 2 ghost points
    prev_values = np.zeros((upper_limit-lower_limit+1,))
    new_values = np.zeros((upper_limit-lower_limit+1,))
    ghost_left = np.zeros((time_steps,))
    ghost_right = np.zeros((time_steps,))
    prev_values[:] = initial_cond
    ghost_left[0] = pipeline[0].recv()
    ghost_right[0] = pipeline[2].recv()
    for i in range(1, time_steps):
        for j in range(1, len(initial_cond)-1):
            # For inner points, normal finite difference scheme
            new_values[j] = fds((prev_values[j-1], prev_values[j], prev_values[j+1]), (delta_x, delta_t))
        # Special handling for delay left; block process if max_delay is
        # exceeded.

        while (pipeline[0].poll() and index_left < i-1) or (i-index_left>max_delay+1):
            ghost_left[index_left+1] = pipeline[0].recv()
            index_left += 1
        new_values[0] = fds((ghost_left[index_left], prev_values[0], prev_values[1]), (delta_x, delta_t))
        pipeline[1].send(new_values[0])
        while (pipeline[2].poll() and index_right < i-1) or (i-index_right>max_delay+1):
            ghost_right[index_right+1] = pipeline[2].recv()
            index_right += 1
        new_values[-1] = fds((prev_values[-2], prev_values[-1], ghost_right[index_right]), (delta_x, delta_t))
        pipeline[3].send(new_values[-1])
        if i == times[times_index]:
            times_index = min(times_index+1, len(times)-1)
            # Need to copy values here with np.array
            result.append([(time()-tic), np.array(new_values)])
        prev_values = new_values
    return result

def get_initial_cond(grid_resolution):
    return list(map(lambda x: amplitude*math.sin(
            wavenumber*2*math.pi*x/grid_resolution + phase_angle),
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

def plot_results(results, delay):
    time_points = list(range(0, time_steps, int((time_steps/1000))))
    grid = [0 for _ in range(len(time_points))]
    time = [[] for _ in range(len(results[0]))]
    for result in results:
        for j, sub_time in enumerate(result):
            time[j].append(sub_time[0])
    for i in range(len(time_points)):
        sub_grid = []
        for j in range(amount_pe):
            # Cache inefficient
            sub_grid.append(results[j][i][-1])
        grid[i] = np.array(sub_grid).reshape((grid_resolution,))
    time = [np.mean(t) for t in time]
    residual = [np.linalg.norm(exact_solution(i)-grid[j], 1) for j, i in enumerate(time_points)]
    print(residual)
    plt.loglog(time, residual, label="Delay " + str(delay))
    #plt.plot(exact_solution(time_points[90])-grid[90], label=delay)

r_alpha = 0.01
grid_resolution = 128
# Choose this so that delta_t/delta_x**2 = 0.1
delta_x = 2*math.pi/grid_resolution
delta_t = r_alpha*delta_x*delta_x
amount_pe = 4
wavenumber = 3
amplitude = 1
phase_angle = 2.5    
alpha = 1
c = 0
time_steps = int(1/(2*math.pi*delta_t))
# Length of each processing element
length_pe = grid_resolution//amount_pe

if __name__ == '__main__':
    plt.clf()
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
    # for the left point, Information from the right point, where to .. right point]
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
# =============================================================================
#     tic = time()
#     for i in range(amount_pe):
#         # Define processes with initial conditions, range, pipelines and delays
# 
#         pe.append(Process(target=solve,
#                           args=((initial_cond[pe_ranges[i][0]:pe_ranges[i][1]+1],
#                                 pe_ranges[i], pipeline[i], 100000,
#                                 [time_steps, delta_x, delta_t]))))
#     [proc.start() for proc in pe]
#     [proc.join(10) for proc in pe]
#     print(time()-tic)
# =============================================================================
    tic = time()
    #x = np.linspace(0.1, 60, 1000)
    #y = x**2
    #plt.loglog(x, y, "--", label="quadratic")
    for delay_time in [0, 1, 10, 100]:
        with Pool(amount_pe) as pool:
            params = [(initial_cond[pe_ranges[i][0]:pe_ranges[i][1]+1],
                       pe_ranges[i], pipeline[i], 1,[time_steps, delta_x,
                                delta_t, r_alpha]) for i in range(amount_pe)]
            results = pool.map(solve, params, 1)
        plot_results(results, delay=delay_time)
    print(time()-tic)
    plt.legend(loc="upper right")
