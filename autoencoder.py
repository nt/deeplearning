import cPickle, gzip, numpy
import theano
import theano.tensor as T
import time, os
from logreg import LogisticRegression

class AutoEncoder(object):
  def __init__(self, numpy_rng, theano_rng=None, input=None, n_visible=784, n_hidden=500, W=None, bhid=None, bvis=None):
    
    self.n_hidden=n_hidden
    self.n_visible=n_visible

    if not theano_rng:
      theano_rng = T.shared_randomstreams.RandomStreams(numpy_rng.randint(2**32))
    self.theano_rng = theano_rng

    if not W:
      initial_W = numpy.asarray(
        numpy_rng.uniform(
          low = -4*numpy.sqrt(6./(n_hidden+n_visible)),
          high = 4*numpy.sqrt(6./(n_hidden+n_visible)),
          size = (n_visible, n_hidden)),
        dtype=theano.config.floatX)
      W = theano.shared(value=initial_W, name='W')

    if not bvis:
      bvis = theano.shared(numpy.zeros(n_visible, 
        dtype=theano.config.floatX), 'bvis')
    
    if not bhid:
      bhid = theano.shared(numpy.zeros(n_hidden, 
        dtype=theano.config.floatX), 'bhid')

    self.W = W
    self.b = bhid
    self.b_prime = bvis
    self.W_prime = self.W.T

    if input == None:
      self.x = T.dmatrix(name='input')
    else:
      self.x = input

    self.params = [self.W, self.b, self.b_prime]

  def get_hidden_values(self, input):
    return T.nnet.sigmoid(T.dot(input, self.W)+self.b)

  def get_reconstructed_input(self, hidden):
    return T.nnet.sigmoid(T.dot(hidden, self.W_prime)+self.b_prime)

  def get_cost_updates(self, corruption_level, learning_rate):
    corrupted_x = self.get_corrupted_input(self.x, corruption_level)
    y = self.get_hidden_values(corrupted_x)
    z = self.get_reconstructed_input(y)
    L = -T.sum(self.x*T.log(z) + (1-self.x)*T.log(1-z), axis=1)
    cost = T.mean(L)
    updates = [(p, p-learning_rate*T.grad(cost, p)) for p in self.params]
    return (cost, updates)

  def get_corrupted_input(self, input, corruption_level):
    return self.theano_rng.binomial(size=input.shape, n=1, p=1-corruption_level)*input

from mnist import *

def train(training_epochs=500, learning_rate=0.1, batch_size=20):
  n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size 

  index = T.lscalar()
  x = T.matrix('x')
  
  states = []

  rng = numpy.random.RandomState(123)
  theano_rng = T.shared_randomstreams.RandomStreams(rng.randint(2**32))

  da = AutoEncoder(numpy_rng=rng, theano_rng=theano_rng, input=x, n_visible=28*28, n_hidden=500)

  cost, updates = da.get_cost_updates(corruption_level=0.2, learning_rate=learning_rate)

  train = theano.function([index], cost, updates=updates,
    givens= { x:train_set_x[index*batch_size:(index+1)*batch_size] })
  
  print train_set_x.get_value().shape
  start_time = time.clock()
  for epoch in xrange(training_epochs):
    c = []
    for batch_index in xrange(n_train_batches):
      c.append(train(batch_index))
    print "Training epoch %d, cost %f" % (epoch, numpy.mean(c))
    states.append([epoch, numpy.mean(c)] + [p.get_value() for p in da.params])
  end_time = time.clock()
  print "training took %f minutes" % ((end_time-start_time)/60)
  
  print "Wrote states to target/autoencoder_states.pkl.gz"
  f = gzip.open('target/autoencoder_states.pkl.gz', 'w')
  cPickle.dump(states, f)
  f.close()

if __name__ == '__main__':
  train(training_epochs=5)
