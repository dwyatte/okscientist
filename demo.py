import nlpfuns
import sys, os, time
import numpy
import scipy.sparse as sparse
import scipy.spatial.distance as distance

# flags
LOAD_VOCAB = False                                      # whether to build vocab (slow) or read from disk
LOAD_FEATURES = False                                   # whether to compute features (slow) or read from disk
# parameters
PDF_ROOT = 'allpdfs'                                    # where to search for PDFs
STOP_FILE = 'stoplist.txt'                              # http://jmlr.csail.mit.edu/papers/volume5/lewis04a/a11-smart-stop-list/english.stop
DOCS_FILE = PDF_ROOT + '_db.txt'                        # flat db of dumpable pdfs -- need to save these to keep indexing into features correct
VOCAB_FILE = PDF_ROOT + '_vocab.json'                   # vocab file to read from/write to disk
FEATURES_FILE_STEM = PDF_ROOT + '_features'             # features file stem name which we build on when saving different kinds of features
FEATURES_FILE_LOAD = PDF_ROOT + '_features_tf_idf.mtx'  # actual features to load
GRAPH_STEM = PDF_ROOT + '_graph'                        # stem of graph file to write
N_REDUCE_FEATURES = 100                                 # dimensionality of reduced features
WEIGHT_THRESH = 0.5                                     # threshold for including an edge in threshold graph function
KNN_K = 5                                               # how many edges (k) should we include out per node in knn graph function

if __name__ == '__main__':

    starttime = time.time()

    # if we are on a Mac, we can use the supplied pdftotext binary
    # TODO: make sure it is in our path, else error
    if sys.platform == 'darwin':
        os.environ['PATH'] = 'bin:' + os.environ['PATH']                      

    # Load vocab from disk or make first pass through corpus: build our vocabulary, which is a python dictionary with
    # word counts (in case we need them for more sophisticated NLP tricks). 
    # also load/save doclist which is a list of the dumpable pdfs
    if LOAD_VOCAB: 
        print '\nReading existing vocabulary %s and associated db' % (VOCAB_FILE)
        docs = nlpfuns.ReadFlatText(DOCS_FILE)
        stoplist = nlpfuns.ReadFlatText(STOP_FILE)
        vocab = nlpfuns.ReadJSON(VOCAB_FILE)
    else:
        print '\nBuilding vocabulary (could take awhile)...'
        docs = nlpfuns.FindPDFs(PDF_ROOT)
        stoplist = nlpfuns.ReadFlatText(STOP_FILE)
        vocab = {}
        docsrm = []                
        for doc,didx in zip(docs, range(len(docs))):
            print 'Adding text from %s (%d/%d)' % (doc, didx+1, len(docs))
            doctext = nlpfuns.DumpPDF(doc, filterjunk=True, stoplist=stoplist)
            # if doc was not dumpable (i.e., image-based pdf, or copyrighted, etc.), rm from list
            if len(doctext) > 0:
                vocab = nlpfuns.UpdateVocab(doctext, vocab)
            else:
                print '|---> Could not dump text from this paper, scheduling for removal\n'
                docsrm.append(doc)
                continue        
        print '\nRemoving %d docs from db' % (len(docsrm))        
        docs = [x for x in docs if x not in docsrm]
        print 'Done. Writing vocabulary %s and associated db\n' % (VOCAB_FILE)
        nlpfuns.WriteFlatText(DOCS_FILE, docs)
        nlpfuns.WriteJSON(VOCAB_FILE, vocab)
    
    
    # Load features from disk, which is a numdocs x numwords (in vocab) sparse matrix to save space.
    # Or compute features and various transformations on them
    if LOAD_FEATURES:
        print 'Loading existing features from %s\n' % (FEATURES_FILE_LOAD)
        features = nlpfuns.ReadMTX(FEATURES_FILE_LOAD)        
    else:
        features = sparse.lil_matrix((len(docs), len(vocab)))    
        print 'Computing features (could take awhile)...'
        for doc,didx in zip(docs, range(len(docs))):
            print 'Computing TF features from %s (%d/%d)' % (doc, didx+1, len(docs))
            doctext = nlpfuns.DumpPDF(doc, filterjunk=True, stoplist=stoplist)            
            features[didx,:] = nlpfuns.ComputeFreqFeatures(doctext, vocab)        
        print '\nDone. Writing TF features to %s\n' % (FEATURES_FILE_STEM + '_tf.mtx')
        nlpfuns.WriteMTX(FEATURES_FILE_STEM + '_tf.mtx', features)

        # convert features to tf-idf (convert features to more efficient sparse matrix type for math ops)
        print 'Computing TF-IDF features...'        
        features = nlpfuns.ComputeTFIDFFeatures(features, vocab)
        print 'Done. Writing TF-IDF features to %s\n' % (FEATURES_FILE_STEM + '_tf_idf.mtx')
        nlpfuns.WriteMTX(FEATURES_FILE_STEM + '_tf_idf.mtx', features)

    print 'Reducing features to %d dimensions...\n' % (N_REDUCE_FEATURES)
    features = nlpfuns.ReduceFeatures(features, N_REDUCE_FEATURES)
    
    print 'Creating graphs and writing to Pajek file...'
    distances = distance.pdist(features, 'cosine')
    distances = distance.squareform(distances)
    
    graphthresh = nlpfuns.CreateGraphThresh(docs, 1-distances, WEIGHT_THRESH)
    graphthresh = nlpfuns.ReduceGraphUndirected(graphthresh)
    # list comprehension just strips off last part of file path for node label
    nlpfuns.WriteGraphPajek(GRAPH_STEM+'_thresh'+str(WEIGHT_THRESH)+'.net', graphthresh, [x.split(os.sep)[-1] for x in docs])
    
    graphknn = nlpfuns.CreateGraphKNN(docs, 1-distances, KNN_K)
    graphknn = nlpfuns.ReduceGraphUndirected(graphknn)
    # list comprehension just strips off last part of file path for node label
    nlpfuns.WriteGraphPajek(GRAPH_STEM+'_knn'+str(KNN_K)+'.net', graphknn, [x.split(os.sep)[-1] for x in docs])

    print 'Done. (elapsed time %f secs)\n' % (time.time() - starttime)