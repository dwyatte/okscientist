import nlpfuns
import sys, os, time
import numpy
import scipy.sparse as sparse
import scipy.spatial.distance as distance

# * try N threshold, write out top N most similar papers -- requires lots more logic though...
#   - need full distance matrix so that we don't get degenerate cases at bottom-right triangle
#   - save all nodes/weights in a dict of dicts {nodesnd : {nodercv : wt; nodercv : wt; ...} }?
#   - then need to filter out symmetric edges 
# for nodercv in dict[nodesnd].keys()
#    if nodercv in dict and dict[nodesnd][key] == dict[nodercv][key]:
#       remove?
#
# try gensim version


# various flags/parameters
LOAD_VOCAB = True                                       # whether to build vocab (slow) or read from disk
LOAD_FEATURES = True                                    # whether to compute features (slow) or read from disk

PDF_ROOT = 'allpdfs'                                    # where to search for PDFs
VOCAB_FILE = 'vocab.json'                               # vocab file to read from/write to disk
STOP_FILE = 'stoplist.txt'                              # stop list file (found online)
FEATURES_FILE_STEM = PDF_ROOT + '_features'             # features file stem name
FEATURES_FILE_LOAD = PDF_ROOT + '_features_tfidf.mtx'   # actual features to load
N_REDUCE_FEATURES = 100                                 # dimensionality of reduced features
WEIGHT_THRESH = 0.75                                    # thresh for writing out an edge in threshold output function
KNN_K = 5                                               # how many edges (k) should we write out in knn output function

if __name__ == '__main__':

    starttime = time.time()

    # if we are on a Mac, we can use the supplied pdftotext binary
    # TODO: make sure it is in our path, else error
    if sys.platform == 'darwin':
        os.environ['PATH'] = 'bin:' + os.environ['PATH']         

    docs = nlpfuns.FindPDFs(PDF_ROOT)
    stoplist = nlpfuns.ReadFlatText(STOP_FILE)               

    # Load vocab from disk or make first pass through corpus: build our vocabulary, which is a python dictionary with
    # word counts (in case we need them for more sophisticated NLP tricks). 
    if LOAD_VOCAB: 
        print '\nReading existing vocabulary from %s' % (VOCAB_FILE)
        vocab = nlpfuns.ReadJSON(VOCAB_FILE)
    else:
        vocab = {}
        print '\nBuilding vocabulary (could take awhile)...'
        for i,doc in enumerate(docs, start=1):
            print 'Adding text from %s (%d/%d)' % (doc, i, len(docs))
            doctext = nlpfuns.DumpPDF(doc, filterjunk=True, stoplist=stoplist)
            # if doc was not dumpable (i.e., image-based pdf, or copyrighted, etc.), rm from list
            if len(doctext) > 0:
                vocab = nlpfuns.UpdateVocab(doctext, vocab)
            else:
                print '|---> Could not dump text from this paper, skipping\n'
                continue
        print '\nDone. Writing vocabulary to %s\n' % (VOCAB_FILE)
        nlpfuns.WriteJSON(VOCAB_FILE, vocab)
 
    
    # Load features from disk, which is a numdocs x numwords (in vocab) sparse matrix to save space.
    # Or compute features and various transformations on them
    if LOAD_FEATURES:
        print 'Loading existing features from %s\n' % (FEATURES_FILE_LOAD)
        features = nlpfuns.ReadMTX(FEATURES_FILE_LOAD)        
    else:
        features = sparse.lil_matrix((len(docs), len(vocab)))    
        print 'Computing features (could take awhile)...'
        for i,doc in enumerate(docs, start=1):
            print 'Computing TF features from %s (%d/%d)' % (doc, i, len(docs))
            doctext = nlpfuns.DumpPDF(doc, filterjunk=True, stoplist=stoplist)
            # if doc was not dumpable (i.e., image-based pdf, or copyrighted, etc.), rm from list
            if len(doctext) > 0:
                vocab = nlpfuns.UpdateVocab(doctext, vocab)
            else:
                print '|---> Could not dump text from this paper, skipping\n'
                continue
            features[i-1,:] = nlpfuns.ComputeFreqFeatures(doctext, vocab)        
        print 'Done. Writing TF features to %s\n' % (FEATURES_FILE_STEM + '_tf.mtx')
        nlpfuns.WriteMTX(FEATURES_FILE_STEM + '_tf.mtx', features)

        # convert features to tf-idf (convert features to more efficient sparse matrix type for math ops)
        print 'Computing TF-IDF features...'        
        features = nlpfuns.ComputeTFIDFFeatures(features.tocsr(), vocab)
        print 'Done. Writing TF-IDF features to %s\n' % (FEATURES_FILE_STEM + '_tf_idf.mtx')
        nlpfuns.WriteMTX(FEATURES_FILE_STEM + '_tf_idf.mtx', features)

    print 'Reducing features to %d dimensions...\n' % (N_REDUCE_FEATURES)
    features = nlpfuns.ReduceFeatures(features, N_REDUCE_FEATURES)
    
    print 'Computing inter-document distance...\n'
    distances = distance.pdist(features, 'cosine')
    distances = distance.squareform(distances)
    
    print 'Writing Pajek graph file...'
    # list comprehension in second arg just gets filename off the path
    nlpfuns.WriteGraphPajekKNN('graph.net', [x.split(os.sep)[-1] for x in docs], 1-distances, KNN_K)

    print 'Done. (elapsed time %f secs)' % (time.time() - starttime)