import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter(object):
    def __init__(self, A = None, H = None, B = None, Q = None, C = None, M = None, s0 = None):

        if(A is None or H is None):
            raise ValueError("Matrices A and H are necessary.")

        self.p = A.shape[1]
        self.m = H.shape[1]

        self.A = A
        self.H = H
        self.B = B
        self.Q = np.eye(self.p) if Q is None else Q
        self.C = np.eye(self.p) if C is None else C
        self.M = np.eye(self.p) if M is None else M
        self.s = np.zeros((self.p, 1)) if s0 is None else s0

    def update(self, z):
        # To be filled out
        # This function takes as an argument the current measurement z[n] and returns the MMSE x[n|n]
        s_predict = np.dot(self.A, self.s)
        P_predict = np.dot(np.dot(self.A, self.C), self.A.T) + self.Q
        
        # Update step (a posteriori)
        z = np.array([z]).reshape(-1, 1) if np.isscalar(z) else z.reshape(-1, 1)
        y = z - np.dot(self.H, s_predict)  # Innovation
        S = np.dot(np.dot(self.H, P_predict), self.H.T) + self.M  # Innovation covariance
        K = np.dot(np.dot(P_predict, self.H.T), np.linalg.inv(S))  # Kalman gain
        
        # Update state and covariance
        self.s = s_predict + np.dot(K, y)
        self.C = P_predict - np.dot(np.dot(K, self.H), P_predict)
        
        return self.s
        


def example():
    # Generate measurements from a continuous true system
	s = np.linspace(-10, 10, 100)
	measurements = - (s**2 + 2*s - 2)  + np.random.normal(0, 2, 100)

    # Discretize system
	dt = 1.0/60 # Sampling time
	A = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
	H = np.array([1, 0, 0]).reshape(1, 3)
	Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
	C = np.array([0.5]).reshape(1, 1)

    # Kalman filter
	kf = KalmanFilter(A = A, H = H, Q = Q, C = C)
	predictions = []
	for z in measurements:
        s_estimate = kf.update(z)
        predictions.append(np.dot(H,  s_estimate)[0])

    # Plots
	plt.plot(range(len(measurements)), measurements, label = 'x[n]')
	plt.plot(range(len(predictions)), np.array(predictions), label = 'x[n|n]')
	plt.legend()
	plt.show()

if __name__ == '__main__':
    example()


# ** Experiments:
# Now you can try to answer, theoretically and experimentally, the following questions:
#
# * What happens when Q is 0 / small / large ?
# * What happens when C is 0 / small / large ?
# * Modify the mean of the noise to a nonzero value. Does the filter still work well? 
# * Replace the measurement generation with a suitably-defined discrete-time system. 
# * Invent a dynamical system with p>>2 and m=1. What happens to the prediction performance?