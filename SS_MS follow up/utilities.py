import uproot as up
import numpy as np
import matplotlib.pyplot as plt
import os




##############################################################################################################
def create_hex_close_packed_array(sensor_radius, array_radius):
    """
    Create a hexagonally close-packed array of circles within a circular boundary.
    The array is first generated as a rectangular hex grid, and points that
    would lie outside the boundary are removed.

    Parameters
    ----------
    sensor_radius : float
        Radius of each small circle.
    array_radius : float
        Radius of the enclosing circle.

    Returns
    -------
    positions : np.ndarray of shape (N, 2)
        Array of (x, y) coordinates of circle centers.
    """
    # hexagonal grid spacing
    dx = 2 * sensor_radius
    dy = np.sqrt(3) * sensor_radius

    # number of rows and columns to cover the circle
    n_rows = int(np.ceil((2 * array_radius) / dy))
    n_cols = int(np.ceil((2 * array_radius) / dx))

    positions = []

    for row in range(n_rows):
        y = -array_radius + row * dy
        if abs(y) > array_radius - sensor_radius:
            continue  # skip rows outside circle
        # offset every other row by one radius
        x_offset = 0.0 if row % 2 == 0 else sensor_radius
        for col in range(n_cols):
            x = -array_radius + col * dx + x_offset
            # keep only if fully inside the circle
            if np.hypot(x, y) <= array_radius - sensor_radius:
                positions.append((x, y))

    return np.array(positions)


##############################################################################################################
def ExtractTGraph(filename):
    with up.open(filename) as infile:
        graphnames = infile.keys()
        if len(graphnames) > 1:
            print('ERROR: more than one graph detected: {}'.format(graphnames))
            return
        graph = infile[graphnames[0]]
    return graph

##############################################################################################################
# Create a function that bins the data as a function of distance and computes the Poisson mean
def BinTGraph(x_in, y_in, bin_width=0.1, x_max=40.):
    bins = np.arange(0., x_max, bin_width)
    x_out = 0.5 * (bins[1:] + bins[:-1])
    digitized = np.digitize(x_in, bins)
    y_out = np.array([y_in[digitized == i].mean()/1.e4 if len(y_in[digitized == i]) > 0 else 0 for i in range(1, len(bins))])
    n = np.array([len(y_in[digitized == i]) for i in range(1, len(bins))])
    err = np.sqrt( (n*1e4) * y_out * (1-y_out) ) / (n*1e4)
    return x_out, y_out, err

##############################################################################################################

def fitfunc(r, A, a, b, alpha, r0):
    rho = r/r0
    return A * np.exp(-a * rho / (1 + rho**(1-alpha)) - b / (1 + rho**(-alpha)))


##############################################################################################################
def simulate_array_response(point, num_photons, sensor_positions, lrf_params, QE = 0.25, spe_resolution = 0.3, saturation_threshold = 10000, saturation_smearing_percent = 0.001):
    responses = []
    for sensor_pos in sensor_positions:
        distance = np.linalg.norm(point - sensor_pos)
        if distance == 0.:
            distance = 0.001  # avoid singularity at r=0
        mean_response = num_photons * QE * fitfunc(distance, *lrf_params)
        response = np.random.poisson(mean_response)
        # Add Gaussian smearing to simulate SPE resolution and other effects
        if response > 0:
            response = np.random.normal(response, np.sqrt(response) * spe_resolution)
        if response > saturation_threshold:
            response = np.random.normal(response, saturation_smearing_percent/100. * response)
        responses.append(response)
    return np.array(responses), np.sum(responses), np.max(responses)