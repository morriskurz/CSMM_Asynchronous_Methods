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
        if i == time_steps-1:
            result = (time()-tic, new_values)
        prev_values = new_values
    return result

def get_initial_cond(grid_resolution):
    return list(map(lambda x: amplitude*math.sin(
            wavenumber*2*math.pi*x/grid_resolution + phase_angle),
            list(range(grid_resolution))))

def exact_solution(time, resolution):
    temp = []
    t = delta_t*time
    for i in range(resolution):
        x = delta_x*i
        sol = math.exp(-alpha*(wavenumber**2)*t)*amplitude*math.sin(
                wavenumber*x+phase_angle-c*t)
        temp.append(sol)
    return temp

def plot_results(results):
    for i, sub in enumerate(results):
        time = []
        grid = []
        for j, point in enumerate(sub):
            time.append(point[0])
            residual = np.linalg.norm(
                    exact_solution(int(1/(2*math.pi*all_delta_t[j])),
                                   grid_resolution[j])-point[1], 2)
            grid.append(residual)
        plt.loglog(time, grid, colors[delays[i]], label=delay_to_label[delays[i]])
    #plt.plot(exact_solution(time_points[90])-grid[90], label=delay)

def setup_pipeline(amount_pe, length_pe, initial_cond, resolution):
    # Using pipelines for communication, each process need 2 receiving
    # pipelines (ghost points) and 2 outgoing pipelines corresponding to
    # adjacent boundary points.
    # Structure: [Receiving information from the left point, where to write information
    # for the left point, Information from the right point, where to .. right point]
    pipeline = [[None, None, None, None] for _ in range(amount_pe)]
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
            boundary = resolution-1
        else:
            # Otherwise use next point
            boundary = (i+1)*length_pe-1
        # Ghost points for boundary points, here the computated
        # information gets stored
        pipeline[i][1].send(initial_cond[i*length_pe])
        #print(pipeline[i-1][2].recv()==initial_cond[i*length_pe])
        pipeline[i][3].send(initial_cond[boundary])
    return pipeline


def setup_pe_ranges(amount_pe, resolution, length_pe):
    pe_ranges = []
    for i in range(amount_pe):
        if i == amount_pe-1:
            # If on the border of the domain, use rightmost point
            boundary = resolution-1
        else:
            # Otherwise use next point
            boundary = (i+1)*length_pe-1
        pe_ranges.append((i*length_pe, boundary))
    return pe_ranges

def combine_results(results, amount_pe, resolution):
    grid = []
    time = []
    for result in results:
        time.append(result[0])
        grid.append(result[1])
    grid = np.array(grid).reshape((resolution,))
    time = np.mean(time)
    return (time, grid)
    
colors = {0: "b.", 1: "g.", 10: "r.", 100: "k.", 1000: "m."}
delay_to_label = {0: "Synchronous", 1: "Delay 1", 10: "Delay 10", 100: "Delay 100", 1000: "Delay 1000"}
r_alpha = 0.1
grid_resolution = [20, 40, 60, 80, 100]
all_delta_x = [2*math.pi/r for r in grid_resolution]
all_delta_t = [r_alpha*x*x for x in all_delta_x]
delays = [0, 1]
amount_pe = 4
wavenumber = 3
amplitude = 1
phase_angle = 0.33   
alpha = 1
c = 0
# Length of each processing element

if __name__ == '__main__':
    plt.clf()
    # Ranges of the processing elements
    pe_ranges = []
    # Processing elements
    pe = []
    tic = time()
    #x = np.linspace(0.1, 60, 1000)
    #y = x**2
    #plt.loglog(x, y, "--", label="quadratic")
    all_results = [[0 for _ in grid_resolution] for _ in delays]
    for i, resolution in enumerate(grid_resolution):
        length_pe = resolution//amount_pe
        delta_x = 2*math.pi/resolution
        delta_t = r_alpha*delta_x*delta_x
        time_steps = int(1/(2*math.pi*delta_t))
        # Get initial condition.
        initial_cond = get_initial_cond(resolution)
        pe_ranges = setup_pe_ranges(amount_pe, resolution, length_pe)
        for j, delay_time in enumerate(delays):
            pipeline = setup_pipeline(amount_pe, length_pe, initial_cond, resolution)
            with Pool(amount_pe) as pool:
                params = [(initial_cond[pe_ranges[i][0]:pe_ranges[i][1]+1],
                           pe_ranges[i], pipeline[i], 1,[time_steps, delta_x,
                                    delta_t, r_alpha]) for i in range(amount_pe)]
                results = pool.map(solve, params, 1)
            results = combine_results(results, amount_pe, resolution)
            all_results[j][i] = results
        print("Run", i, "Time:", time()-tic)
    print(all_results)
    plot_results(all_results)
    print(time()-tic)
    plt.legend(loc="upper right")
