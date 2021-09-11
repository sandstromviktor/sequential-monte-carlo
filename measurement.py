import numpy as np 

class Measurement():
    def __init__(self, mc, stations, eta = 3, xi = 1.5, nu = 90):
        self.mc = mc
        self.stations = stations
        self.num_stations = stations.shape[1]
        self.eta = eta
        self.xi = xi
        self.nu = nu

    def take_measurement(self):
        state = next(self.mc)
        x1 = state[0,:]
        x2 = state[3,:]
        station_x1 = self.stations[0,:]
        station_x2 = self.stations[1,:]
        dx1 = (np.repeat(x1,self.num_stations) - np.tile(station_x1, self.mc.num_particles))
        dx2 = (np.repeat(x2,self.num_stations) - np.tile(station_x2, self.mc.num_particles))
        d = np.linalg.norm(np.array([dx1,dx2]),axis=0).reshape(self.mc.num_particles,self.num_stations)
        noise = np.random.normal(0,self.xi,size=(self.mc.num_particles, self.num_stations))

        return state, self.nu - 10*self.eta*np.log10(d) + noise
         
