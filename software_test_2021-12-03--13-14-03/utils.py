#!/usr/bin/python3

import numpy as np
from scipy.signal import savgol_filter

def filter_black_regions(start, final, count, critical_depth = 200.):
    # Maybe instead of returning the maximum value, find the ones
    # which are in the first areas

    # Use max inds first
    inds = np.argsort(count)[::-1]
    
    for i in inds:
        if start[i] < critical_depth:
            # Found largest black region below critical depth to start from
            return start[i], final[i], count[i]

    # Nothing is below the critical depth, therefore just return initial pixel values
    print("\n filter_black_regions: WARNING: No black region found below critical depth, setting to critical depth")

    return critical_depth, critical_depth, 0

def calculate_black_regions(line, critical_depth=100, critical=False):
    low, high = np.min(line), np.max(line)
    black_regions = np.zeros(line.shape)
    count = 0
    max_count = 0
    start = [0]
    final = [0]
    count = [0]


    indices = np.where(line == low)[0]

    for j, ind in enumerate(indices):
        if j > 0:
            if ind == indices[j-1] + 1:
                # Then two indices are contiguous
                final[-1] = ind
                count[-1] += 1
            else:
                # Not contiguous, start again
                start.append(ind)
                final.append(ind)
                count.append(0)

    if critical:
        return filter_black_regions(start, final, count, critical_depth = critical_depth )
    else:
        return start[0], final[0], count[0]

def calculate_boundary(image, sample_rate=10, offset = 0., n_sigma=3.):
    # Sample from the top

    # Here assuming reasonably high threshold for the analysis,
    # such that all bakelite is black


    max_pixel = np.max(image)
    height, width = image.shape

    n_samples = int(width/float(sample_rate)) 


    surface_x = []
    surface_y = []
    index = 0
    idx_sample = 0
    for i in range(width):
        if i % sample_rate == 0:                
            # Count the pixels which are black

            # Find the maximum, continuous portion of black in image in top
            start, depth, size = calculate_black_regions(image[:,i], critical_depth=0.2*image.shape[1])

            if depth < 0.2 * height:
                # Count it as black region:
                surface_x.append(i)
                surface_y.append(depth)

            else:
                depth = 0
                for j in range(height):
                    if np.isclose(image[j,i], max_pixel, 1e-4):
                        break
                    else:
                        depth += 1
                surface_x.append(i)
                surface_y.append(depth)
            idx_sample += 1

    surface_coordinates = np.zeros((idx_sample, 2))
    surface_coordinates[:,0] = np.asarray(surface_x)
    surface_coordinates[:,1] = np.asarray(surface_y)    
    # Now calculate the statistics using the width of black pixels found.
    sigma = np.std(surface_coordinates[:,1])
    y_mean = np.mean(surface_coordinates[:,1])

    surface_coordinates[ :, 1] += n_sigma * sigma + offset

    surface_coordinates[:,1] = savgol_filter(surface_coordinates[:,1], 21, 3)
    return surface_coordinates, y_mean, sigma
