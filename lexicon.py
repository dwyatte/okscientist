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
	vocab = []	

	doc1 = ['everything', 'was', 'beautiful', 'and', 'nothing', 'hurt']
	vocab = UpdateVocab(doc1, vocab)	
	
	doc2 = ['and', 'so', 'it', 'goes']
	vocab = UpdateVocab(doc2, vocab)
	
	doc3 = ['everything', 'and', 'nothing']
	vocab = UpdateVocab(doc2, vocab)
	
	features = []
	doclist = [doc1, doc2, doc3]
	for doc in doclist:
		features.append(BuildFeatures(doc, vocab))
		
	distances = distance.pdist(features, 'cosine')
	distances = distance.squareform(distances)
	
	print distances