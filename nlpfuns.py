''' Various natural language processing and utility functions -- 
    parsing pdfs, building the vocabulary, features, etc. '''

import subprocess, os, fnmatch, json
import scipy.io as io
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import numpy

##########################################################################################
# Basic file I/O (text reading/writing, json reading/writing)
##########################################################################################

''' Reads flat text file from disk, split by newlines, returns as list '''
def ReadFlatText(file):
    with open(file, 'r') as f:
         return f.read().split()        
              
''' Writes keys part of dictionary to disk (one word per line ignoring values) '''
def WriteFlatText(file, dict):
    with open(file, 'w') as f:
        [f.write(x + '\n') for x in dict.keys()]
    return True
    
''' Writes dictionary to disk in JSON format '''
def WriteJSON(file, dict):
    with open(file, 'wb') as f:
        json.dump(dict, f, indent=1)
    return True
    
''' Reads dictionary from disk in JSON format and returns it '''
def ReadJSON(file):
    with open(file, 'rb') as f:
        return json.load(f)

''' Write data in matrix market format '''
def WriteMTX(file, matrix):
    with open(file, 'wb') as f:
        io.mmwrite(f, matrix)
    return True
    
    ''' Write data in matrix market format '''
def ReadMTX(file):
    with open(file, 'rb') as f:
        return io.mmread(f)

##########################################################################################
# Scraping/parsing functions (PDFs, etc.)
##########################################################################################

''' Find all PDFs from root dir, return list of paths '''
def FindPDFs(root):
    paths = []    
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in fnmatch.filter(filenames, '*.pdf'):
            paths.append(os.path.join(dirpath, filename))  
    return paths 

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
    
        
##########################################################################################
# NLP logic such as vocabulary building, feature computation, etc.
##########################################################################################

''' Update the vocab with words (usually extracted from a pdf), and return new vocab '''
def UpdateVocab(words, vocab):
    for word in words:
        if word in vocab:
            vocab[word] += 1	
        else:
            vocab[word] = 1	
    return vocab

''' Build a word freq feature vector from a list of words (usually extracted from a pdf), according
	to the overall vocabulary '''
def ComputeFreqFeatures(words, vocab):
    freqs = dict.fromkeys(vocab, 0)
    for word in words:
        if word in vocab:
            freqs[word] +=1
            
    # put into numpy array to allow math
    fvals = numpy.array(freqs.values())
    # convert to frequencies
    return fvals/float(numpy.max(fvals))

''' Compute TF-IDF features from all frequency features. freqs is a sparse matrix and vocab is a dict 
    All math in this function is done on numpy arrays (sparse when possible) for efficiency reasons (no dicts) '''
def ComputeTFIDFFeatures(freqs, vocab):
    # compute inverse document frequency
    IDF = numpy.zeros(len(vocab))
    for widx in range(0, len(vocab)):
        IDF[widx] = numpy.log2(freqs.shape[0] / float(1+len(freqs[:,widx].nonzero())))

    # now do TF * IDF
    TF_IDF = sparse.lil_matrix(freqs.shape)
    for didx in range(0, freqs.shape[0]):
        # have to convert sparse array out to dense array for element-wise multiplication (boo...)
        TF_IDF[didx,:] = numpy.multiply(freqs[didx,:].todense(), IDF)
    return TF_IDF

''' Reduce features with dimensionality reduction. Currently only SVD is supported for efficiency '''
def ReduceFeatures(features, dimensions):
    [U, S, V] = linalg.svds(features, dimensions)
    return U*S

##########################################################################################
# Graph file I/O (Pajek format, etc.)
##########################################################################################
        
''' Writes out graph in pajek format. Uses similarities as edge weights, which should
    be symmetrical, so just write upper right triangle of matrix. Weights is a numpy
    array. Thresh is a threshold that weight must surpass to get written as an edge '''
def WriteGraphPajek(netfile, nodes, weights, thresh=0.0):
    with open(netfile, 'w') as f:
        # write node ids
        f.write('*Vertices ' + str(len(nodes)) + '\n')
        for nodeid,nodelabel in enumerate(nodes, start=1):
            f.write(str(nodeid) + ' ' + '\"' + nodelabel + '\"\n')

        # write the weights
        f.write('*Edges\n')    
        for l1idx in range(0, len(nodes)):
            for l2idx in range(l1idx+1, len(nodes)):
                if weights[l1idx, l2idx] > thresh:
                    f.write(str(l1idx+1) +  ' ' + str(l2idx+1) + ' ' + str(weights[l1idx, l2idx]) + '\n')
    return True