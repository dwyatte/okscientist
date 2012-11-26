''' Various natural language processing and utility functions -- 
    parsing pdfs, building the vocabulary, features, etc. '''

import subprocess, os, fnmatch
import numpy

''' Loads and returns flat text file from disk, split by newlines '''
def ReadFlatText(file):
    with open(file, 'r') as f:
         return f.read().split()        
         
''' Writes list to disk (flat text, one word per line) '''
def WriteFlatText(file, list):
    with open(file, 'w') as f:
        [f.write(x + '\n') for x in list]
        
''' Writes out graph in pajek format. Uses distances as edge weights, which should
    be symmetrical, so just write upper write triangle of matrix. Requires nupmy '''
def WriteGraphPajek(netfile, labels, weights):
    with open(netfile, 'w') as f:
        f.write('*vertices ' + str(len(labels)) + '\n')
        for nodeid,label in enumerate(labels):
            f.write(str(nodeid) + '\t' + '\"' + label + '\"\n')
        
        f.write('*arcs\n')    
        for l1idx in range(0, len(labels)):
            for l2idx in range(l1idx+1, len(labels)):
                f.write(str(l1idx) +  '\t' + str(l2idx) + '\t' + str(weights[l1idx, l2idx]) + '\n')
            
''' Update the vocab with words (usually extracted from a pdf), and return new vocab '''
def UpdateVocab(words, vocab):
	for word in words:
		if word not in vocab:
			vocab.append(word)
	return vocab


''' dump contents of pdf at path using the binary pdftotext (assume binary is in path). '''
def DumpPDF(path, filterjunk, stoplist):
    # third arg ('-') denotes that we are dumping the text to stdout, so we need to redirect
    pdftotextProc = subprocess.Popen(['pdftotext', path, '-'], stdout=subprocess.PIPE)        
    (stdoutdata, sterrdata) = pdftotextProc.communicate()
    stdoutdata = stdoutdata.lower().split()
    
    # do some basic filtering -- only return words that are all letters with length > 1
    if filterjunk:
        stdoutdata = [x for x in stdoutdata if x.isalpha() and len(x) > 1]
    
    # do more specific filtering according to stoplist
    if len(stoplist) > 0:
        stdoutdata = [x for x in stdoutdata if x not in stoplist]
    
    return stdoutdata

	
''' Find all PDFs from root dir, return list of paths '''
def FindPDFs(root):
    paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in fnmatch.filter(filenames, '*.pdf'):
            paths.append(os.path.join(dirpath, filename))  
    return paths  


''' Build a feature vector from a list of words (usually extracted from a pdf), according
	to the overall vocabulary '''
def BuildFeatures(words, vocab):
	features = dict.fromkeys(vocab, 0)
	for word in words:
	    if word in vocab:
    		features[word] +=1
	return features.values()