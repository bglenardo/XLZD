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