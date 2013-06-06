import cPickle, gzip, numpy
import theano
import theano.tensor as T
from mnist import *
import time, os

class LogisticRegression(object):
  def __init__(self, input, n_in, n_out):
    self.b = theano.shared(numpy.zeros((n_out,)), name='b')
    self.W = theano.shared(numpy.zeros((n_in,n_out)), name='W')
    self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W)+self.b)
    self.y_pred = T.argmax(self.p_y_given_x, axis=1)

  def negative_log_likelihood(self, y):
    return - T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

  def errors(self, y):
    return T.mean(T.neq(self.y_pred, y))



def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                           batch_size=600):
  
  # compute number of minibatches for training, validation and testing
  n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
  n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
  n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

  print "Building model"

  index = T.lscalar('index') #index of mini batch
  x = T.matrix('x')
  y = T.ivector('y')

  classifier = LogisticRegression(
            input=x.reshape((batch_size, 28 * 28)), 
            n_in=28 * 28, 
            n_out=10)

  cost = classifier.negative_log_likelihood(y)

  test_model = theano.function(
    inputs = [index],
    outputs = classifier.errors(y),
    givens = {
      x: test_set_x[index*batch_size : (index+1)*batch_size],
      y: test_set_y[index*batch_size : (index+1)*batch_size]
    }
  )

  validate_model = theano.function(
    inputs = [index],
    outputs = classifier.errors(y),
    givens = {
      x: valid_set_x[index*batch_size : (index+1)*batch_size],
      y: valid_set_y[index*batch_size : (index+1)*batch_size]
    }
  )

  # Train

  g_W = T.grad(cost, classifier.W)
  g_b = T.grad(cost, classifier.b)

  updates = [(classifier.W, classifier.W - learning_rate*g_W),
            (classifier.b, classifier.b - learning_rate*g_b)]

  train_model = theano.function(
    inputs = [index],
    outputs = cost,
    updates = updates,
    givens = {
      x: train_set_x[index*batch_size : (index+1)*batch_size],
      y: train_set_y[index*batch_size : (index+1)*batch_size]
    }
  )

  print "Training model"
  patience = 5000
  patience_increase = 2

  improvement_threshold = 0.995

  validation_frequency = min(n_train_batches, patience/2)

  best_params = None
  best_validation_loss = numpy.inf
  test_score = 0.
  start_time = time.clock()

  done_looping = False
  epoch = 0

  states = []

  while (epoch < n_epochs) and (not done_looping):
    epoch += 1
    for minibatch_index in xrange(n_train_batches):
      minibatch_avg_cost = train_model(minibatch_index)
      #iteration number
      iter = (epoch-1)*n_train_batches + minibatch_index
      if((iter+1) % validation_frequency):
        validation_losses = [validate_model(i)
                              for i in xrange(n_valid_batches)]
        this_validation_loss = numpy.mean(validation_losses)
        print('epoch %i, minibatch %i/%i, validation error %f %%' % \
            (epoch, minibatch_index + 1, n_train_batches,
            this_validation_loss * 100.))
        if this_validation_loss < best_validation_loss * improvement_threshold:
          patience = max(patience, iter+patience_increase)
          best_validation_loss = this_validation_loss
          test_losses = [test_model(i) for i in xrange(n_test_batches)]
          test_score = numpy.mean(test_losses)
          states.append((iter, best_validation_loss, test_score, classifier.W.get_value(), classifier.b.get_value()))
          print(('     epoch %i, minibatch %i/%i, test error of best'
                       ' model %f %%') %
                        (epoch, minibatch_index + 1, n_train_batches,
                         test_score * 100.))
      if patience <= iter:
        done_looping = True
        break

  end_time = time.clock()
  print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
        (best_validation_loss * 100., test_score * 100.))
  print 'The code ran for %d epochs, with %f epochs/sec' % (
    epoch, 1. * epoch / (end_time - start_time))
  print ('The code for file ' +
    os.path.split(__file__)[1] +
    ' ran for %.1fs' % ((end_time - start_time)))

  print "Wrote states to target/logreg_states.pkl.gz"
  f = gzip.open('target/logreg_states.pkl.gz', 'w')
  cPickle.dump(states, f)
  f.close()

if __name__ == '__main__':
  sgd_optimization_mnist()
