import nlpfuns
import sys, os, subprocess
import numpy
import scipy.spatial.distance as distance
import matplotlib.pyplot as plot

# various flags/parameters
DOC_ROOT = '/Users/dwyatte/Documents/Papers/'   # where to search for PDFs
BUILD_VOCAB = 0                                 # whether to build vocab (slow) or read from disk
VOCAB_FILE = 'vocab.dat'                        # vocab file to read from/write to disk
WEIGHT_THRESH = 0.5                             # threshold for writing out an edge in output function

if __name__ == '__main__':

    # if we are on a Mac, we can use the supplied pdftotext binary
    # TODO: make sure it is in our path, else error
    if sys.platform == 'darwin':
        os.environ['PATH'] = 'bin:' + os.environ['PATH']         
        
    # gather up our doc paths 
    docs = nlpfuns.FindPDFs(DOC_ROOT)
        
        
    if BUILD_VOCAB:
        vocab = []
        stoplist = nlpfuns.ReadFlatText('stoplist.dat')    
        print '\nBuilding vocabulary (could take awhile)...\n'
        for i,doc in enumerate(docs, start=1):
            print 'Adding text from %s (%d/%d)' % (doc, i, len(docs))
            doctext = nlpfuns.DumpPDF(doc, filterjunk=True, stoplist=stoplist)
            # if doc was not dumpable (i.e., image-based pdf, or copyrighted, etc.), rm from list
            if len(doctext) > 0:
                vocab = nlpfuns.UpdateVocab(doctext, vocab)
            else:
                print '|---> Could not dump text from this paper, skipping\n'
                continue
        nlpfuns.WriteFlatText(VOCAB_FILE, vocab)
    else:
        print '\nReading existing vocabulary from %s\n' % (VOCAB_FILE)
        vocab = nlpfuns.ReadFlatText(VOCAB_FILE)
        stoplist = nlpfuns.ReadFlatText('stoplist.dat')    
 
    
    # compute features, node labels
    features = [] 
    labels = []
    print '\nComputing features (could take awhile)...'
    for i,doc in enumerate(docs, start=1):
        print 'Computing features from %s (%d/%d)' % (doc, i, len(docs))
        doctext = nlpfuns.DumpPDF(doc, filterjunk=True, stoplist=stoplist)
        # if doc was not dumpable (i.e., image-based pdf, or copyrighted, etc.), rm from list
        if len(doctext) > 0:
            vocab = nlpfuns.UpdateVocab(doctext, vocab)
        else:
            print '|---> Could not dump text from this paper, skipping\n'
            continue
        f = nlpfuns.BuildFeatures(doctext, vocab)
        f = numpy.array(f)
        # normalize
        f = (f-numpy.min(f)/(numpy.max(f)-numpy.min(f)))
        features.append(f)
        labels.append(doc.split(os.sep)[-1])      
        	    
    distances = distance.pdist(features, 'cosine')
    distances = distance.squareform(distances)

    nlpfuns.WriteGraphPajek('graph.net', labels, 1-distances, WEIGHT_THRESH)