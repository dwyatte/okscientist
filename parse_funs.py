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
	return features
	
	
if __name__ == '__main__':
	words1 = ['everything', 'was', 'beautiful', 'and', 'nothing', 'hurt']
	vocab = []	
	vocab = UpdateVocab(words1, vocab)	
	
	words2 = ['and', 'so', 'it', 'goes']
	vocab = UpdateVocab(words2, vocab)	
	
	features = BuildFeatures(words2, vocab)
	print features
	print features.values()