import nlpfuns
import sys, os
import networkx
import numpy
import scipy.spatial.distance as distance
import numpy.linalg as linalg
import matplotlib.pyplot as plot

#########################################################################################
# TODO:
# * make a BuildVocab wrapper in nlpfuns that returns/writes to disk
# * figure out how to deal with bad pdfs -- probably just monitor stderr
#########################################################################################

if __name__ == '__main__':
    vocab = []
    stoplist = nlpfuns.LoadStoplist('english.stop')
    
    # if we are on a mac, we can use the supplied pdftotext binary
    if sys.platform == 'darwin':
        os.environ['PATH'] = 'bin:' + os.environ['PATH']
        
    # gather up our docs from the test directory
    #docs = nlpfuns.FindPDFs('test')
    docs = nlpfuns.FindPDFs('/Users/dwyatte/Documents/Papers/2010/')
        
    # dump text and build vocab
    print 'Building vocabulary (could take awhile)...'
    for i,doc in enumerate(docs):
        doctext = nlpfuns.DumpPDF(doc, filterjunk=True, stoplist=stoplist)
        vocab = nlpfuns.UpdateVocab(doctext, vocab)
        print '%d/%d' % (i, len(docs))
    
    # compute features
    features = [] 
    for doc in docs:
        print 'Computing features for %s' % doc
        doctext = nlpfuns.DumpPDF(doc, filterjunk=True, stoplist=stoplist)
        f = nlpfuns.BuildFeatures(doctext, vocab)
        f = numpy.array(f)
        # normalize
        f = (f-numpy.min(f)/(numpy.max(f)-numpy.min(f)))
        features.append(f)        
	    
    distances = distance.pdist(features, 'cosine')
    distances = distance.squareform(distances)
    
    plot.Figure()
    plot.imshow(distances)
    plot.show()