''' Various natural language processing and utility functions -- 
    parsing pdfs, building the vocabulary, features, etc. '''

import subprocess, os, fnmatch


''' Loads and returns flat text file from disk, split by newlines '''
def ReadFlatText(file):
    with open(file, 'r') as f:
         return f.read().split()        
       
         
''' Writes list to disk (flat text, one word per line) '''
def WriteFlatText(file, list):
    with open(file, 'w') as f:
        [f.write(x + '\n') for x in list]
    return True
       
        
''' Writes out graph in pajek format. Uses similarities as edge weights, which should
    be symmetrical, so just write upper right triangle of matrix. Note that weights is a 
    numpy array. thresh is a threshold that weight must surpass to get written as an edge '''
def WriteGraphPajek(netfile, labels, weights, thresh=0.0):
    with open(netfile, 'w') as f:
        # write node ids
        f.write('*vertices ' + str(len(labels)) + '\n')
        for nodeid,label in enumerate(labels, start=1):
            f.write(str(nodeid) + ' ' + '\"' + label + '\"\n')

        # write the weights
        f.write('*Edges\n')    
        for l1idx in range(0, len(labels)):
            for l2idx in range(l1idx+1, len(labels)):
                if weights[l1idx, l2idx] > thresh:
                    f.write(str(l1idx+1) +  ' ' + str(l2idx+1) + ' ' + str(weights[l1idx, l2idx]) + '\n')
  
    return True
            
            
''' Update the vocab with words (usually extracted from a pdf), and return new vocab '''
def UpdateVocab(words, vocab):
	for word in words:
		if word not in vocab:
			vocab.append(word)
	
	return vocab


''' dump contents of pdf at path using the binary pdftotext (assume binary is in path). '''
def DumpPDF(path, filterjunk, stoplist):
    # third arg ('-') denotes that we are dumping the text to stdout, so we need to redirect
    pdftotextProc = subprocess.Popen(['pdftotext', path, '-'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)        
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