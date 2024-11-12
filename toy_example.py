import numpy as np
np.random.seed(0)

# the function we want to optimize
def f(w):
  def quadratic(z):
        return np.sum(z**2)
  return -quadratic(w)

dim = 2
# hyperparameters
npop = 50 # population size
sigma = 0.1 # noise standard deviation
alpha = 0.001 # learning rate

# start the optimization
solution = np.zeros(dim) # our initial guess is that the solution is at the origin
w = np.random.randn(dim) # our initial guess is random
for i in range(1000):

  # print current fitness of the most likely parameter setting
  if i % 100 == 0:
    print('iter %d. w: %s, solution: %s, reward: %f' % 
          (i, str(w), str(solution), f(w)))

  # initialize memory for a population of w's, and their rewards
  N = np.random.randn(npop, dim) # samples from a normal distribution N(0,1)
  R = np.zeros(npop)
  for j in range(npop):
    w_try = w + sigma*N[j] # jitter w using gaussian of sigma 0.1
    R[j] = f(w_try) # evaluate the jittered version

  # standardize the rewards to have a gaussian distribution
  A = (R - np.mean(R)) / np.std(R)
  # perform the parameter update. The matrix multiply below
  # is just an efficient way to sum up all the rows of the noise matrix N,
  # where each row N[j] is weighted by A[j]
  w = w + alpha/(npop*sigma) * np.dot(N.T, A)

# print final fitness of the most likely parameter setting
print('final w: %s, solution: %s, reward: %f' % (w, solution,f(w)))