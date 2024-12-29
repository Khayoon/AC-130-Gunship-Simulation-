import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Conversion constants
FEET_TO_METERS = 0.3048
KNOTS_TO_MS = 0.514444

# Physics constants
g = 9.81
rho = 1.225
omega_earth = 7.292115e-5
lat = np.radians(45)

class ProjectileSimulation:
    def __init__(self, 
                 mass=45.36,
                 diameter=0.155,
                 aircraft_speed=110,
                 aircraft_altitude=10000,
                 firing_angle=-45.0,
                 wind_speed=5.0,
                 wind_angle=0.0,
                 delay_distance=2000):  # Distance in feet before firing
        
        self.aircraft_altitude = aircraft_altitude * FEET_TO_METERS
        self.aircraft_speed = aircraft_speed * KNOTS_TO_MS
        self.delay_distance = delay_distance * FEET_TO_METERS
        self.firing_position = np.array([self.delay_distance, self.aircraft_altitude, 0])
        
        self.m = mass
        self.d = diameter
        self.A = np.pi * (diameter/2)**2
        self.Cd = 0.47
        
        self.theta = np.radians(firing_angle)
        self.wind_speed = wind_speed
        self.wind_angle = np.radians(wind_angle)
        self.Cl = 0.0001
        
        # Initialize simulation parameters
        self.reset_simulation()
    
    def reset_simulation(self):
        muzzle_velocity = 150
        self.y0 = np.array([
            self.delay_distance,    # Start x at delay distance
            self.aircraft_altitude, # y position (altitude)
            0.0,                    # z position
            self.aircraft_speed + muzzle_velocity * np.cos(self.theta),
            muzzle_velocity * np.sin(self.theta),
            0.0
        ])
    
    def air_density(self, altitude):
        return 1.225 * np.exp(-altitude/7400)
    
    def forces(self, t, state):
        x, y, z, vx, vy, vz = state
        
        rho_h = self.air_density(y)
        v_rel = np.array([vx, vy, vz]) - self.wind_speed * np.array([np.cos(self.wind_angle), 0, np.sin(self.wind_angle)])
        v_mag = np.linalg.norm(v_rel)
        
        if v_mag > 0:
            drag_coef = -0.6 * self.Cd * rho_h * self.A * v_mag
            drag = drag_coef * v_rel / self.m
            
            spin_vector = np.array([0, 0.1, 0])
            magnus = (self.Cl * rho_h * self.A * np.cross(spin_vector, v_rel)) / self.m
            
            coriolis_scale = 0.1
            coriolis = coriolis_scale * 2 * omega_earth * np.array([
                -vz * np.sin(lat),
                0,
                vx * np.sin(lat)
            ])
            
            ax = drag[0] + magnus[0] + coriolis[0]
            ay = drag[1] + magnus[1] - g
            az = drag[2] + magnus[2] + coriolis[2]
        else:
            ax = 0
            ay = -g
            az = 0
        
        return np.array([vx, vy, vz, ax, ay, az])
    
    def simulate(self, t_max=60.0):
        def hit_ground(t, y):
            return y[1]
        hit_ground.terminal = True
        hit_ground.direction = -1
        
        solution = solve_ivp(
            self.forces,
            (0, t_max),
            self.y0,
            method='RK45',
            events=hit_ground,
            rtol=1e-8,
            atol=1e-8
        )
        return solution.t, solution.y

class ProjectileAnimation:
    def __init__(self, simulation):
        self.sim = simulation
        self.t, self.trajectory = self.sim.simulate()
        
        # Pre-launch idle flight time (3 seconds)
        self.pre_launch_time = 3
        self.pre_launch_frames = int(self.pre_launch_time * 20)  # 20 fps
        
        # Calculate aircraft positions before launch
        self.aircraft_positions = self.calculate_aircraft_positions()
        
        self.fig = plt.figure(figsize=(15, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.setup_plot()
        
        self.line, = self.ax.plot([], [], [], 'b-', label='Trajectory')
        self.point, = self.ax.plot([], [], [], 'ro', label='Projectile')
        self.aircraft, = self.ax.plot([], [], [], 'k^', label='AC-130', markersize=10)
        
    def calculate_aircraft_positions(self):
        # Generate aircraft positions for pre-launch
        t = np.linspace(0, self.pre_launch_time, self.pre_launch_frames)
        x = self.sim.aircraft_speed * t
        y = np.full_like(t, self.sim.aircraft_altitude)
        z = np.zeros_like(t)
        return np.vstack((x, y, z))
    
    def setup_plot(self):
        margin = 1.1
        max_x = max(np.max(self.trajectory[0]), self.sim.delay_distance) / FEET_TO_METERS
        max_y = np.max(self.trajectory[1]) / FEET_TO_METERS
        max_z = np.max(np.abs(self.trajectory[2])) / FEET_TO_METERS
        
        self.ax.set_xlim(0, max_x * margin)
        self.ax.set_ylim(-max_z * margin, max_z * margin)
        self.ax.set_zlim(0, max_y * margin)
        
        self.ax.set_xlabel('Distance (feet)')
        self.ax.set_ylabel('Lateral Distance (feet)')
        self.ax.set_zlabel('Altitude (feet)')
        self.ax.set_title('AC-130 Projectile Trajectory')
        
        self.ax.grid(True)
    
    def update(self, frame):
        if frame < self.pre_launch_frames:
            # aircraft moving
            aircraft_x = self.aircraft_positions[0, frame] / FEET_TO_METERS
            aircraft_y = self.aircraft_positions[2, frame] / FEET_TO_METERS
            aircraft_z = self.aircraft_positions[1, frame] / FEET_TO_METERS
            
            self.aircraft.set_data([aircraft_x], [aircraft_y])
            self.aircraft.set_3d_properties([aircraft_z])
            
            self.line.set_data([], [])
            self.line.set_3d_properties([])
            self.point.set_data([], [])
            self.point.set_3d_properties([])
        else:
            # Post-launch aircraft and projectile
            proj_frame = frame - self.pre_launch_frames
            if proj_frame < len(self.trajectory[0]):
                x = self.trajectory[0,:proj_frame] / FEET_TO_METERS
                y = self.trajectory[2,:proj_frame] / FEET_TO_METERS
                z = self.trajectory[1,:proj_frame] / FEET_TO_METERS
                
                self.line.set_data(x, y)
                self.line.set_3d_properties(z)
                
                if proj_frame > 0:
                    self.point.set_data(x[-1:], y[-1:])
                    self.point.set_3d_properties(z[-1:])
                
                # Update aircraft position (continuing movement)
                aircraft_x = (self.sim.delay_distance + self.sim.aircraft_speed * 
                            (self.t[min(proj_frame, len(self.t)-1)])) / FEET_TO_METERS
                self.aircraft.set_data([aircraft_x], [0])
                self.aircraft.set_3d_properties([self.sim.aircraft_altitude / FEET_TO_METERS])
        
        return self.line, self.point, self.aircraft
    
    def animate(self):
        total_frames = self.pre_launch_frames + len(self.t)
        anim = FuncAnimation(
            self.fig, 
            self.update,
            frames=total_frames,
            interval=50,
            blit=True
        )
        plt.legend()
        plt.show()
        return anim

if __name__ == "__main__":
    sim = ProjectileSimulation(
        mass=45.36,
        diameter=0.155,
        aircraft_speed=110,
        aircraft_altitude=10000,
        firing_angle=-45,
        wind_speed=5,
        wind_angle=0,
        delay_distance=2000  # Distance in feet before firing
    )
    
    anim = ProjectileAnimation(sim)
    anim.animate()