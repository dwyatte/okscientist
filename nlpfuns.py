'''Various natural language processing and utility functions -- 
    parsing pdfs, building the vocabulary, features, etc.'''

import subprocess

stoplist = []

'''Loads stoplist into global variable from file'''
def LoadStoplist(stopfile):
    global stoplist    
    with open(stopfile, 'r') as f:
         stoplist = f.read().split()
    return True


'''Update the vocab with words (usually extracted from a pdf), and return new vocab.
	assumes words have been filtered according to a blacklist of common words and
	all punctuation and case has already been removed'''
def UpdateVocab(words, vocab):
	for word in words:
		if word not in vocab:
			vocab.append(word)
	return vocab
	

'''TODO: Save Vocab to disk'''
def SaveVocab(vocab):
    pass


'''TODO: Load Vocab from disk'''
def LoadVocab():
    pass
	

'''Build a feature vector from a list of words (usually extracted from a pdf), according
	to the overall vocabulary'''
def BuildFeatures(words, vocab):
	features = dict.fromkeys(vocab, 0)
	for word in words:
		features[word] +=1
	return features.values()
	
	
''' dump contents of pdf filename using the binary pdftotext (path supplied as arg2)'''
def DumpPDF(filename, pdftotextPath):
    # third arg of - denotes that we are dumping the text to stdout, so we need to redirect
    pdftotext_proc = subprocess.Popen([pdftotextPath, filename, '-'], stdout=subprocess.PIPE)
    pdftotext_proc.wait()
    (stdoutdata, stderrdata) = pdftotext_proc.communicate()
    
    stoplist = LoadStoplist('english.stop')

    # do some basic filtering
    stdoutdata = stdoutdata.split()
    stdoutdata = [x for x in stdoutdata if x.isalpha()]
    return FilterText(stdoutdata, stoplist)


''' Filter text through a stoplist '''
def FilterText(text, stoplist):
    return [x for x in text if x not in stoplist]