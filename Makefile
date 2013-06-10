pre-build:
	mkdir -p target

clean:
	rm -rf target

target/mnist.pkl.gz:
	wget http://deeplearning.net/data/mnist/mnist.pkl.gz -O target/mnist.pkl.gz

target/logreg_states.pkl.gz: target/mnist.pkl.gz
	python logreg.py

target/logreg_001.png: target/logreg_states.pkl.gz
	python logreg_visualize.py

target/logreg.gif: target/logreg_001.png
	convert -delay 15 -loop 1 target/logreg_*.png target/logreg.gif

target/autoencoder_states.pkl.gz: target/mnist.pkl.gz
  python autoencoder.py

all: target/logreg.gif
