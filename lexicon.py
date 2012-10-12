import itertools
import scipy.spatial.distance as distance
import numpy 

'''Update the vocab with words (usually extracted from a pdf), and return new vocab.
	assumes words have been filtered according to a blacklist of common words and
	all punctuation and case has already been removed'''
def UpdateVocab(words, vocab):
	for word in words:
		if word not in vocab:
			vocab.append(word)
	return vocab
	

'''Build a feature vector from a list of words (usually extracted from a pdf), according
	to the overall vocabulary'''
def BuildFeatures(words, vocab):
	features = dict.fromkeys(vocab, 0)
	for word in words:
		features[word] +=1
	return features.values()
	
	
if __name__ == '__main__':

	doc1 = ['everything', 'was', 'beautiful', 'and', 'nothing', 'hurt']
	doc2 = ['and', 'so', 'it', 'goes']
	doc3 = ['everything', 'and', 'nothing']
	doclist = [doc1, doc2, doc3]
	
	# build the vocab over all docs
	vocab = []	
	for doc in doclist:
		vocab = UpdateVocab(doc, vocab)	
	
	# compute features
	features = []
	for doc in doclist:
		f = BuildFeatures(doc, vocab)
		f = numpy.array(f)
		# normalize
		f = (f-numpy.min(f))/(numpy.max(f)-numpy.min(f))
		features.append(f)
		
	distances = distance.pdist(features, 'cosine')
	distances = distance.squareform(distances)
	
	print distances