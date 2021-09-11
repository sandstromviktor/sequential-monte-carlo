# Sequential Monte Carlo method

In this notebook we will develop a sequential Monte Carlo method for filtering out the path of some target by collecting measurements from different locations. This could for example be a moving cell phone where different antennas are picking up different signal strengths.
The motion of the target is modelled as a Markov chain, and with the measurements, we form a hidden Markov model (HMM) and the goal is to observ measurements and filter out the path using sequential importance sampling. 

See notebook!
