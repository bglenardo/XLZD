import numpy as np

# This is a code for simulating the performance of a fisheye lens
# for imaging events in dual-phase xenon TPC. The basic setup involves
# a 5mm-thick surface that emits light from straight lines perpendicular
# to the surface. Our problem is to create simulated images of these lines
# on a sensor, and to determine how well we can image events of various shapes.

# This code should create a class that stores data about the optical setup

# How do I want to pass an object to the imaging system? Probably I want to pass a grid of points that represent
# voxels in 3D space, each with a weight that represents the amount of light emitted from that voxel. 

# So, the fisheye class should have a method that takes in a grid of points and weights, and produces
# an image on the sensor. The grid of points will be in the 3D space of the object plane.

class Fisheye:
    def __init__(self, distance_to_object_cm=7., field_of_view_radius_cm=30., projection_type='stereographic'):
        
        self.distance_to_object_cm = distance_to_object_cm
        self.field_of_view_radius_cm = field_of_view_radius_cm
        
        self.field_of_view_theta = np.arctan(field_of_view_radius_cm / distance_to_object_cm)
        
        self.projection_type = projection_type
        
        self.allowed_projection_types = ['equidistant', 'stereographic', 'orthographic', 'equisolid_angle']
        if self.projection_type not in self.allowed_projection_types:
            raise ValueError(f"Invalid projection type: {self.projection_type}. Allowed types are: {self.allowed_projection_types}")
        
        self.sensor = None
        self.focal_length_cm = None
        self.object_points_3d = None
        self.object_weights_3d = None
        self.image_points_2d_mm = None
        self.sensor_image = None

    def PrintInfo(self):
        print(f"Distance to object: {self.distance_to_object_cm:.2f} cm")
        print(f"Field of view radius: {self.field_of_view_radius_cm:.2f} cm")
        print(f"Field of view angle: {np.degrees(self.field_of_view_theta):.2f} degrees")
        print(f"Projection type: {self.projection_type}")
        
        
    def DefineSensor(self, pixel_size_mm=0.01, num_pixels_x=1024, num_pixels_y=1024):
        self.sensor = Sensor(pixel_size_mm, num_pixels_x, num_pixels_y)
        self.sensor.PrintInfo()
        
    def CalculateFocalLength(self):
        # The focal length calculation is determined by the number that maps the edge of the desired
        # field of view to the edge of the sensor.
        if self.projection_type == 'equidistant':
            self.focal_length_mm = self.sensor.max_image_radius_mm / np.radians(self.field_of_view_theta)
        elif self.projection_type == 'stereographic':
            self.focal_length_mm = self.sensor.max_image_radius_mm / (2 * np.tan(self.field_of_view_theta / 2))
        elif self.projection_type == 'orthographic':
            self.focal_length_mm = self.sensor.max_image_radius_mm / np.sin(self.field_of_view_theta)
        elif self.projection_type == 'equisolid_angle':
            self.focal_length_mm = self.sensor.max_image_radius_mm / (2 * np.sin(self.field_of_view_theta / 2))
        
        
    def DefineObject(self, object_points_3d, object_weights_3d):
        self.object_points_3d = object_points_3d
        self.object_weights_3d = object_weights_3d

    def CreatePhotonSourceFromTrack(self, track_points_xyE, photons_per_energy=1.0,
                                    kernel_sigma_um=100.0, z_min_mm=0.0, z_max_mm=5.0,
                                    count_mode='deterministic', rng_seed=None,
                                    normalize_negative_energy='clip'):
        # Build photon origins from (x, y, E) track points.
        track_points_xyE = np.asarray(track_points_xyE, dtype=float)
        if track_points_xyE.ndim != 2 or track_points_xyE.shape[1] != 3:
            raise ValueError("track_points_xyE must be an Nx3 array with columns (x_cm, y_cm, E).")

        if z_max_mm < z_min_mm:
            raise ValueError("z_max_mm must be greater than or equal to z_min_mm.")

        if photons_per_energy < 0:
            raise ValueError("photons_per_energy must be non-negative.")

        if kernel_sigma_um < 0:
            raise ValueError("kernel_sigma_um must be non-negative.")

        allowed_count_modes = ['deterministic', 'poisson']
        if count_mode not in allowed_count_modes:
            raise ValueError(f"Invalid count_mode: {count_mode}. Allowed values are: {allowed_count_modes}")

        allowed_negative_energy_modes = ['clip', 'raise']
        if normalize_negative_energy not in allowed_negative_energy_modes:
            raise ValueError(
                f"Invalid normalize_negative_energy: {normalize_negative_energy}. "
                f"Allowed values are: {allowed_negative_energy_modes}"
            )

        valid_mask = np.all(np.isfinite(track_points_xyE), axis=1)
        valid_rows = track_points_xyE[valid_mask]

        if valid_rows.size == 0:
            self.DefineObject(np.empty((0, 3), dtype=float), np.empty((0,), dtype=float))
            return {
                'photon_points_3d_cm': self.object_points_3d,
                'photon_weights': self.object_weights_3d,
                'metadata': {
                    'n_input_points': int(track_points_xyE.shape[0]),
                    'n_valid_points': 0,
                    'n_generated_photons': 0,
                    'count_mode': count_mode,
                    'kernel_sigma_um': float(kernel_sigma_um),
                    'z_range_mm': [float(z_min_mm), float(z_max_mm)],
                    'rng_seed': rng_seed,
                }
            }

        x_cm = valid_rows[:, 0]
        y_cm = valid_rows[:, 1]
        energies = valid_rows[:, 2]

        if normalize_negative_energy == 'raise' and np.any(energies < 0):
            raise ValueError("Negative energies found in track_points_xyE while normalize_negative_energy='raise'.")

        energies = np.clip(energies, 0.0, None)
        expected_counts = energies * photons_per_energy

        if count_mode == 'deterministic':
            photon_counts = np.rint(expected_counts).astype(int)
        else:
            rng = np.random.default_rng(rng_seed)
            photon_counts = rng.poisson(expected_counts).astype(int)

        total_photons = int(np.sum(photon_counts))
        if total_photons == 0:
            self.DefineObject(np.empty((0, 3), dtype=float), np.empty((0,), dtype=float))
            return {
                'photon_points_3d_cm': self.object_points_3d,
                'photon_weights': self.object_weights_3d,
                'metadata': {
                    'n_input_points': int(track_points_xyE.shape[0]),
                    'n_valid_points': int(valid_rows.shape[0]),
                    'n_generated_photons': 0,
                    'count_mode': count_mode,
                    'kernel_sigma_um': float(kernel_sigma_um),
                    'z_range_mm': [float(z_min_mm), float(z_max_mm)],
                    'rng_seed': rng_seed,
                }
            }

        sigma_cm = kernel_sigma_um * 1e-4
        z_min_cm = z_min_mm * 0.1
        z_max_cm = z_max_mm * 0.1

        # Expand source points according to sampled photon count per track point.
        x_rep = np.repeat(x_cm, photon_counts)
        y_rep = np.repeat(y_cm, photon_counts)

        rng_xy = np.random.default_rng(rng_seed)
        if sigma_cm > 0:
            photon_x_cm = x_rep + rng_xy.normal(loc=0.0, scale=sigma_cm, size=total_photons)
            photon_y_cm = y_rep + rng_xy.normal(loc=0.0, scale=sigma_cm, size=total_photons)
        else:
            photon_x_cm = x_rep
            photon_y_cm = y_rep

        photon_z_cm = rng_xy.uniform(z_min_cm, z_max_cm, size=total_photons) + self.distance_to_object_cm

        photon_points_3d_cm = np.column_stack((photon_x_cm, photon_y_cm, photon_z_cm))
        photon_weights = np.ones(total_photons, dtype=float)

        self.DefineObject(photon_points_3d_cm, photon_weights)

        return {
            'photon_points_3d_cm': photon_points_3d_cm,
            'photon_weights': photon_weights,
            'metadata': {
                'n_input_points': int(track_points_xyE.shape[0]),
                'n_valid_points': int(valid_rows.shape[0]),
                'n_generated_photons': total_photons,
                'count_mode': count_mode,
                'kernel_sigma_um': float(kernel_sigma_um),
                'z_range_mm': [float(z_min_mm), float(z_max_mm)],
                'rng_seed': rng_seed,
            }
        }
        
    def ProjectPoints(self):
        if self.object_points_3d is None:
            raise ValueError("Object points have not been defined. Please call DefineObject() first.")
        object_points_3d = self.object_points_3d
        
        
        # This method will take in a grid of points in 3D space, and project them onto the sensor plane using the specified projection type.
        # The points_3d should be an Nx3 array of (x, y, z) coordinates in the object plane.
        
        # First, I need to compute the angle theta for each point, 
        # which is the angle between the point and the optical axis.
        thetas = np.arctan2(np.sqrt(object_points_3d[:, 0]**2 + object_points_3d[:, 1]**2), object_points_3d[:, 2])
        # Then, I need to compute the radius on the sensor plane for each point, based on the projection type.
        if self.projection_type == 'equidistant':
            radii_mm = self.focal_length_mm * thetas
        elif self.projection_type == 'stereographic':
            radii_mm = 2 * self.focal_length_mm * np.tan(thetas / 2)
        elif self.projection_type == 'orthographic':
            radii_mm = self.focal_length_mm * np.sin(thetas)
        elif self.projection_type == 'equisolid_angle':
            radii_mm = 2 * self.focal_length_mm * np.sin(thetas / 2)
            
        # Finally, I need to convert the radii to coordinates on the sensor plane.
        # The angle phi is the angle in the x-y plane, which can be computed as:
        phis = np.arctan2(object_points_3d[:, 1], object_points_3d[:, 0])
        # The coordinates on the sensor plane are then:
        x_coords = radii_mm * np.cos(phis)
        y_coords = radii_mm * np.sin(phis)
        # Make image_points_2d_cm a Nx2 array
        self.image_points_2d_mm = np.column_stack((x_coords, y_coords))
        
        
    def ProduceImage(self):
        # This method will take the projected points and create an image on the sensor plane.
        # Basically it maps the 2D coordinates to pixel coordinates, then creates an image based
        # on the weights of the points.
        if self.image_points_2d_mm is None:
            raise ValueError("Image points have not been calculated. Please call ProjectPoints() first.")
    
        # Convert the 2D coordinates from cm to mm
        image_points_2d_mm = self.image_points_2d_mm
        # Convert the 2D coordinates from mm to pixel coordinates
        pixel_coords_x = (image_points_2d_mm[:, 0] + self.sensor.array_size_x / 2) / self.sensor.pixel_size_mm
        pixel_coords_y = (image_points_2d_mm[:, 1] + self.sensor.array_size_y / 2) / self.sensor.pixel_size_mm
        # Now we have the pixel coordinates, we can create an image based on the weights of the points.
        image = np.zeros((self.sensor.num_pixels_y, self.sensor.num_pixels_x))
        for i in range(len(self.object_weights_3d)):
            x = int(pixel_coords_x[i])
            y = int(pixel_coords_y[i])
            if 0 <= x < self.sensor.num_pixels_x and 0 <= y < self.sensor.num_pixels_y:
                image[y, x] += self.object_weights_3d[i]
        self.sensor_image = image

class Sensor:
    def __init__(self, pixel_size_mm=0.01, num_pixels_x=1024, num_pixels_y=1024):
        self.pixel_size_mm = pixel_size_mm
        self.num_pixels_x = num_pixels_x
        self.num_pixels_y = num_pixels_y
        
        self.array_size_x = self.pixel_size_mm * self.num_pixels_x
        self.array_size_y = self.pixel_size_mm * self.num_pixels_y
        
        self.max_image_radius_mm = min(self.array_size_x, self.array_size_y) / 2
        
        
    def PrintInfo(self):
        print(f"Array size: {self.array_size_x:.2f} mm x {self.array_size_y:.2f} mm")
        print(f"Max image radius: {self.max_image_radius_mm:.2f} mm")
        print(f"Pixel size: {self.pixel_size_mm:.2f} mm")
        print(f"Number of pixels: {self.num_pixels_x} x {self.num_pixels_y}")