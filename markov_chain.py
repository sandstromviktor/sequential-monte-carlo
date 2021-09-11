import numpy as np


class MarkovChain():
    def __init__(self, iter, num_particles, dt, sigma=0.5):
        
        self.state = np.random.normal(np.zeros(6), np.sqrt(np.array([500,5,5,200,5,5])), size=(num_particles,6)).T
        self.num_particles = num_particles
        self.iter = iter
        self.dt = dt
        self.sigma = sigma
        self.n = 0
        self.Phi, self.Psi_z, self.Psi_w, self.transition_matrix = self.get_matricies()
        self.mc_states = np.array([[0,0],[0,3.5],[3.5,0],[0,-3.5],[-3.5,0]])
        self.mc_state = np.random.randint(0,self.mc_states.shape[1], size=(self.num_particles))

    def __repr__(self):
        repr = '\nMarkovChain Generator Object \n \n'
        repr += 'STATE SHAPE: \t \t' + str(self.state.shape) + '\n' + 'MC STATE SHAPE: \t' + str(self.mc_state.shape)
        repr += '\n' + 'Number of particles: \t' + str(self.num_particles) + '\n' 
        repr += '\nParameters: \nNoise std: \t' + str(self.sigma) + '\nTime step \t' + str(self.dt) +'\n \n'
        repr += 'Using transition matrix \n' +  str(self.transition_matrix) 
        return repr

    def __iter__(self):
       return self

    def __next__(self):
        n = self.n
        if n >= self.iter:
            raise StopIteration()

        noise = np.random.normal(0,self.sigma, size=(self.num_particles,2))
        for i,state in enumerate(self.mc_state):
            self.mc_state[i] = int(np.random.choice(np.arange(0,5), 1,  p = self.transition_matrix[state,:].ravel()))
        self.state = self.Phi@self.state + self.Psi_z@self.mc_states[self.mc_state.ravel(),:].T + self.Psi_w@noise.T
        self.n += 1

        return self.state

    def get_matricies(self):
        psi_z = np.array([self.dt**2/2, self.dt, 0])
        psi_w = np.array([self.dt**2/2, self.dt, 1])
        phi = np.array([np.flip(psi_w),[0, 1, self.dt],[0,0,0.6]])
        Phi = np.block([[phi, np.zeros((3,3))],[np.zeros((3,3)),phi]])
        Psi_z = np.block([[psi_z.reshape(-1,1), np.zeros((3,1))],[np.zeros((3,1)), psi_z.reshape(-1,1)]])
        Psi_w = np.block([[psi_w.reshape(-1,1), np.zeros((3,1))],[np.zeros((3,1)), psi_w.reshape(-1,1)]])
        transition_matrix = np.array([[16,1,1,1,1],[1,16,1,1,1],[1,1,16,1,1],[1,1,1,16,1],[1,1,1,1,16]])/20
        return Phi, Psi_z, Psi_w, transition_matrix

'''
import matplotlib.pyplot as plt

num_particles = 20
dt = 0.5
iter = 2000
markov_chain=MarkovChain(iter,num_particles, dt)
x=[]
y=[]
for i in range(iter):
    state = next(markov_chain)
    x.append(state[0,:])
    y.append(state[3,:])

plt.plot(x,y)
plt.show()'''