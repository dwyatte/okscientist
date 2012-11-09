import nlpfuns
import sys, os
import networkx
import numpy
import scipy.spatial.distance as distance

if __name__ == '__main__':
    vocab = []
    stoplist = nlpfuns.LoadStoplist('english.stop')
    
    # if we are on a mac, we can use the supplied pdftotext binary
    if sys.platform == 'darwin':
        os.environ['PATH'] = 'bin:' + os.environ['PATH']
        
    # gather up our docs from the test directory
    docs = nlpfuns.FindPDFs('test')
        
    # dump text and build vocab
    for doc in docs:
        doctext = nlpfuns.DumpPDF(doc, filterjunk=True)
        vocab = nlpfuns.UpdateVocab(doctext, stoplist, vocab)
    
    # compute features
    features = [] 
    for doc in docs:
        doctext = nlpfuns.DumpPDF(doc, filterjunk=True)
        f = nlpfuns.BuildFeatures(doctext, vocab)
        f = numpy.array(f)
        # normalize
        f = (f-numpy.min(f)/(numpy.max(f)-numpy.min(f)))
        features.append(f)
	    
    distances = distance.pdist(features, 'cosine')
    distances = distance.squareform(distances)
    print distances
