# Im Zusammenhang mit der Seminararbeit über
# Asynchronous finite difference schemes for partial differential equations,
# a paper by Diego A. Donzis and Konduri Aditya
# Thomas Camminady googlen für Zusammenfassung der Analysen und MDI
# Sequentielle Ausfuehrung zur Analyse der Auswirkung des Delays
# bzw. der Prozessoranzahl auf die Ordnung des Verfahrens.
#

# TODO: 
# Abspeicherung des Ergebnisses
# plotten
# delay

# Ziel: Vergleich zwischen synchronem Zeitverbrauch und asynchronen mit verschiedenen Delays
# 1. Wahl der Datenstruktur
# 2. synchroner Fall

import numpy as np
import matplotlib
import time
import matplotlib.pyplot as plt
import math
import multiprocessing

time_steps = 10000
grid_resolution = 256
delta_x = 2*math.pi/grid_resolution
delta_t = 10e-5
amount_pe = 4
delay = 1
alpha = 1
c = 0

# Put the points on the border into a dictionary
boundary_points_left = dict()
boundary_points_right = dict()
length_pe = grid_resolution//amount_pe
for i in range(amount_pe):
    if i == amount_pe-1:
        boundary = grid_resolution-1
    else:
        boundary = (i+1)*length_pe-1
    boundary_points_left[i*length_pe] = (i*length_pe-1) % (grid_resolution)
    boundary_points_right[boundary] = (boundary+1) % (grid_resolution)

grid = np.zeros((time_steps, grid_resolution))
grid_sync = np.zeros((time_steps, grid_resolution))
grid_pointer = np.zeros((time_steps, grid_resolution))

# Initial conditions: u(x, 0) = sum over K(A(K) * sin(K*x + phi(K))
# where K is the wavenumber, A(K) amplitude and phi(K) phase angle
wavenumber = 2
# randomly chosen
amplitude = 10
phase_angle = 0.5

# generate initial conditions on a domain of length 2*pi
grid[0, :] = list(map(
    lambda x: amplitude*math.sin(wavenumber*2*math.pi*x/grid_resolution + phase_angle),
    list(range(grid_resolution))))
grid_sync[0, :] = grid[0, :]

def fds(time_steps):
    plt.ion()
    for i in range(1, time_steps):
        for j in range(grid_resolution):
            grid[i, j] = solve(i, j)
            grid_sync[i, j] = solve(i, j, use_delay=False)
        # Plot every 100 time steps
        if i % 100 == 1:
            plot_all(i)
        


def solve(time, point, use_delay=True):
    time = time-1
    if use_delay:
        delay_left, delay_right = 0, 0
        # If it is a boundary point, use a delay
        try:
            boundary_points_left[point]
            delay_left = delay
        except KeyError:
            pass
        try:
            boundary_points_right[point]
            delay_right = delay
        except KeyError:
            pass
    
        time_r = max(time-delay_right, 0)
        time_l = max(time-delay_left, 0)
        temp = alpha*(grid[time_r, (point+1) % grid_resolution]
                    - 2*grid[time, point] + grid[time_l, (point-1)]) / (delta_x**2)
        temp2 = c*(grid[time_r, (point+1) % grid_resolution]
                    - grid[time_l, point-1]) / (2*delta_x)
        return delta_t*(temp-temp2) + grid[time, point]
    else:
        temp = alpha*(grid_sync[time, (point+1) % grid_resolution]
                    - 2*grid_sync[time, point] + grid_sync[time, (point-1)]) / (delta_x**2)
        temp2 = c*(grid_sync[time, (point+1) % grid_resolution]
                    - grid_sync[time, point-1]) / (2*delta_x)
        return delta_t*(temp-temp2) + grid_sync[time, point]

def exact_solution(time):
    temp = []
    t = delta_t*time
    for i in range(grid_resolution):
        x = delta_x*i
        sol = math.exp(-alpha*(wavenumber**2)*t)*amplitude*math.sin(
                wavenumber*x+phase_angle-c*t)
        temp.append(sol)
    return temp


def plot_all(time):
    plt.clf()
    test = exact_solution(time)
    difference = list(map(lambda x: (x[0]-x[1]), zip(grid[time], test)))
    difference_sync = list(map(lambda x: (x[0]-x[1]), zip(grid_sync[time], test)))
    t = np.arange(0., 2*math.pi, 2*math.pi/grid_resolution)
    plt.subplot(1, 2, 1)
    plt.axis([0, 2*math.pi, -10, 10])
    plt.plot(t, test, "r", label="Exact")
    plt.plot(t, grid[time], "b", label="Numeric (asynchronous)")
    plt.plot(t, grid_sync[time], "black", label="Numeric (synchronous)")
    #plt.plot(t, difference, "g", label="Difference")
    plt.legend(loc="upper right")
    #plt.draw()
    #plt.pause(0.001)
    plt.subplot(1, 2, 2)
    #plt.clf()
    plt.axis([0, 2*math.pi, -0.01, 0.01])
    plt.plot(t, difference, "c--", label="Difference (asynchronous)" )
    plt.plot(t, difference_sync, "b--", label="Difference (synchronous)")
    plt.legend(loc="upper right")
    #plt.draw()
    plt.pause(0.0001)
