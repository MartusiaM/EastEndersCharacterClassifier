# EastEndersCharacterClassifier


Tested with Python version - 3.6

1. NLTK

2. sklearn

3. Parse Trees
Stanford CoreNLP from pycorenlp has been used for extracting parse trees. 
This requires setting up and running a server. Details can be found here:
https://stanfordnlp.github.io/CoreNLP/index.html

Starting a server:
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

4. PoS Tags
CRFSuite has been used for extracting CRF PoS Tags. It can be set up by running:
pip install sklearn-crfsuite

5. Tika for parsing pdfs

6. Pickle for saving/loading objects

FOR RUNNING:
1. Create an obj folder in the root of your project
2. The preprocess script expects a folder named "EastEnders_2008_1350-1399" on the
   root. This directory must have two subdirectories called "Train" and "Test", 
   containing the train and test pdf scripts. The train folder used for training had 
   the first 45 pdf files, and the test folder had the remaining five.
3. Run the prepocess script. This will create the required pickle files used by 
   the classifier script.
4. Run the data_frameconversion script 
5. Go through the classifier.py file
6. Based on which features you enable, you might have to set up a few things.
	# Julian's CRF PosTagger expects the crfpostagger file in specified path
	# Glove word2vec expects glove.6B in the path, which can be downloaded from:
	https://nlp.stanford.edu/projects/glove/
	# Parse trees rquire CoreNLP