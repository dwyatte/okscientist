import nlpfuns
import sys, os
import numpy
import scipy.sparse as sparse
import scipy.spatial.distance as distance

# pca/svd
#   http://glowingpython.blogspot.com/2011/07/principal-component-analysis-with-numpy.html
#   http://scicomp.stackexchange.com/questions/2391/derive-pca-with-svd
# try gensim version


# various flags/parameters
DOC_ROOT = 'allpdfs/'                           # where to search for PDFs
BUILD_VOCAB = False                             # whether to build vocab (slow) or read from disk
VOCAB_FILE = 'vocab.json'                       # vocab file to read from/write to disk
COMPUTE_FEATURES = False                        # whether to compute features (slow) or read from disk
FEATURES_FILE = 'features.mtx'                  # features file to read from/write to disk
STOP_FILE = 'stoplist.txt'                      # stop list file (found online)
WEIGHT_THRESH = 0.5                             # threshold for writing out an edge in output function

if __name__ == '__main__':

    # if we are on a Mac, we can use the supplied pdftotext binary
    # TODO: make sure it is in our path, else error
    if sys.platform == 'darwin':
        os.environ['PATH'] = 'bin:' + os.environ['PATH']         

    docs = nlpfuns.FindPDFs(DOC_ROOT)
    stoplist = nlpfuns.ReadFlatText(STOP_FILE)               

    # First pass through corpus: build our vocabulary, which is a python dictionary with
    # word counts (in case we need them for more sophisticated NLP tricks). Alternatively, 
    # we can just load existing vocabulary from disk    
    if BUILD_VOCAB:
        vocab = {}
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
        nlpfuns.WriteJSON(VOCAB_FILE, vocab)
    else:
        print '\nReading existing vocabulary from %s' % (VOCAB_FILE)
        vocab = nlpfuns.ReadJSON(VOCAB_FILE)
 
    
    # Compute features, which is a numdocs x numwords (in vocab) sparse matrix to save space
    if COMPUTE_FEATURES:
        features = sparse.lil_matrix((len(docs), len(vocab)))    
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
            features[i-1,:] = nlpfuns.ComputeFreqFeatures(doctext, vocab)
        # convert features to tf-idf and compute distance
        # (convert features to more efficient sparse matrix type for math ops)
        print '\nComputing TF-IDF features...'        
        features = nlpfuns.ComputeTFIDFFeatures(features.tocsr(), vocab)            	    
        nlpfuns.WriteMTX(FEATURES_FILE, features)
    else:
        print 'Loading existing features from %s\n' % (FEATURES_FILE)
        features = nlpfuns.ReadMTX(FEATURES_FILE)        
    
    print 'Computing inter-document distance...'
    distances = distance.pdist(features.todense(), 'cosine')
    distances = distance.squareform(distances)
    
    print 'Writing Pajek graph file...'
    # list comprehension in second arg just gets filename off the path
    nlpfuns.WriteGraphPajek('graph.net', [x.split(os.sep)[-1] for x in docs], 1-distances, WEIGHT_THRESH)
    print 'Done'