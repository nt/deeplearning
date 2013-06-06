import cPickle, gzip, numpy
from PIL import Image, ImageDraw

def build_visualisation(src='target/logreg_states.pkl.gz'):
  # Load the dataset
  f = gzip.open(src, 'rb')
  states = cPickle.load(f)
  f.close()

  def color(val, min, max):
    mean = (min + max)/2
    if val <= mean:
      return (0, 0, int(2*255*(mean-val)/(max-min)))
    else:
      return (int(2*255*(val-mean)/(max-min)), 0, 0)


  Ws = [state[3] for state in states]
  max = numpy.max(Ws)
  min = numpy.min(Ws)

  for state in range(len(states)):
    tile_s = 6
    w = (28+2)*tile_s*5 + 2*tile_s
    h = (28+2)*tile_s*2 + 20

    im = Image.new('RGB', (w,h))
    draw = ImageDraw.Draw(im)

    iter, validation, test, W, b = states[state]

    for label in range(10):
      w = numpy.reshape(numpy.transpose(W)[label], (28, 28))
      x_offset = (28+2)*tile_s*(label%5)
      y_offset = (28+2)*tile_s*(int(label/5)) + 20

      for i in range(28):
        for j in range(28):
          c = color(w[j][i], min, max)
          draw.rectangle([
            (x_offset+(i+1)*tile_s, y_offset+(j+1)*tile_s),
            (x_offset+(i+2)*tile_s, y_offset+(j+2)*tile_s)],
            fill = c)

    for i in range(10):
      draw.rectangle([
        ((28+2)*tile_s*5, tile_s*(i+1)),
        ((28+2)*tile_s*5+tile_s, tile_s*(i+2))
        ], fill = color(b[i], min, max))

    draw.text((5, 5), "iter=%i test_error=%.2f%% validation_error=%.2f%%" % (iter, test*100, validation*100))

    im.save('target/logreg_%03d.png' % state, 'png')


if __name__ == '__main__':
  print "Building a visualisation of Logistic Regression learning"
  build_visualisation()