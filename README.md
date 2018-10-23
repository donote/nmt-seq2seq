# [Sequence to sequence model for neural machine translation](https://github.com/ZeweiChu/nmt-seq2seq)

### requirements: 
- pytorch 0.3.1
- python 3.6
- pytorch 
- nltk
- tqdm

### How to use
- To train the model, simply run
	./run.sh
- To test the model
	.run_test.sh
- To see what config options you have
	python main.py --help


### background 
- this repo tries to implement the paper [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215), some details differ from the original paper, though. 
- we are using data from http://www.manythings.org/anki/

### TODO
- add attention to the currently very basic model


### Bug report
If you find any bugs, please feel free to send an email to zeweichu@gmail.com , I will try to be responsive!
