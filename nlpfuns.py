''' Various natural language processing and utility functions -- 
    parsing pdfs, building the vocabulary, features, etc. '''

import subprocess, os, fnmatch, json
import scipy.io as io
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import numpy

##########################################################################################
# Basic file I/O (text reading/writing, json reading/writing, matrix market, etc.)
##########################################################################################

''' Reads flat text file from disk, split by newlines, returns as list '''
def ReadFlatText(file):
    with open(file, 'r') as f:
         return f.read().splitlines()
              
''' Writes list to disk (one word per line) '''
def WriteFlatText(file, elements):
    with open(file, 'w') as f:       
        [f.write(x + '\n') for x in elements]
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
    freqs = freqs.tocsr() # csr matrix allows for faster math
    IDF = numpy.zeros(len(vocab))
    for widx in range(0, len(vocab)):
        IDF[widx] = numpy.log2(freqs.shape[0] / float(1+len(freqs[:,widx].nonzero()[0])))
    # as a hack, diagonalize IDF to allow for element-wise matrix multiplication in sparse mode
    IDF = sparse.diags(IDF,0).tocsr() # csr matrix allows for faster math
        
    # now do TF * IDF
    TF_IDF = sparse.lil_matrix(freqs.shape)
    for didx in range(0, freqs.shape[0]):
        TF_IDF[didx,:] = freqs[didx,:] * IDF # IDF is a sparse diagonal matrix
    return TF_IDF

''' Reduce features with dimensionality reduction. Currently only SVD is supported for efficiency '''
def ReduceFeatures(features, dimensions):
    assert dimensions < features.shape[0], 'Reduced feature dimensionality must be less than #observations'
    [U, S, V] = linalg.svds(features, dimensions)
    return U

##########################################################################################
# Graph logic (Pajek output, etc.)
##########################################################################################
    
''' Create a graph, which is just a dict of dicts of the form {node_i : {node_j1: weight_j1; node_j2: weight_j2; ...}}
    thresh specifies the threshold that the weight needs to surpass to get added to the graph. '''
def CreateGraphThresh(nodes, weights, thresh=0.0):
    graph = {}
    for sndnodeid in range(len(nodes)):
        rcvdict = {}
        for rcvnodeid in range(len(nodes)):            
            if sndnodeid != rcvnodeid and weights[sndnodeid,rcvnodeid] > thresh: # no self edges, must exceed thresh
                rcvdict[str(rcvnodeid)] = weights[sndnodeid,rcvnodeid]
        graph[str(sndnodeid)] = rcvdict
    return graph
    
''' Create a graph, which is just a dict of dicts of the form {node_i : {node_j1: weight_j1; node_j2: weight_j2; ...}}
    k specifies value to use in a knn search over similarities '''
def CreateGraphKNN(nodes, weights, k):
    assert k < len(nodes), 'k must be less than #nodes-1'
    
    graph = {}
    for sndnodeid in range(len(nodes)):
        rcvdict = {}
        # sort the weights and used the sorted indices to get the nearest k nodes
        weightssortidxs = numpy.argsort(1-weights[sndnodeid]) # 1-weights to turn them into ascending distances
        kweightsidxs = weightssortidxs[1:1+k] # start at 1 because 0 will always be same paper (most similar to self)
        for kweightidx in kweightsidxs:
            rcvdict[str(kweightidx)] = weights[sndnodeid,kweightidx]
        graph[str(sndnodeid)] = rcvdict
    return graph

''' Reduce a directed graph to an undirected one by removing reciprocal connections between nodes '''
def ReduceGraphUndirected(graph):
    for sndnodeid in graph.keys():
        for rcvnodeid in graph[sndnodeid].keys():
            if sndnodeid in graph[rcvnodeid].keys():
                del graph[rcvnodeid][sndnodeid]
    return graph

''' Writes out graph in pajek format. Begins with *Vertices, which is a nodeid to label mapping,
    then includes *Edges which is the format sndnode rcvnode weight.'''
def WriteGraphPajek(netfile, graph, nodelabels):
    with open(netfile, 'w') as f:
        # write node ids
        f.write('*Vertices ' + str(len(nodelabels)) + '\n')
        for nodelabel,nodeid in zip(nodelabels, range(len(nodelabels))):
            f.write(str(nodeid+1) + ' ' + '\"' + nodelabel + '\"\n')

        # write the weights
        f.write('*Edges\n')    
        for sndnodeid in graph.keys():
            for rcvnodeid in graph[sndnodeid].keys():
                # need to cast to int, make 1-based and cast back to str
                f.write(str(int(sndnodeid)+1) + ' ' + str(int(rcvnodeid)+1) + ' ' + str(graph[sndnodeid][rcvnodeid]) + '\n')
    return True