import nlpfuns
import sys, os, time
import numpy
import scipy.sparse as sparse
import scipy.spatial.distance as distance

# * there is probably some sort of bug where docs that got skipped over are getting written out to graph.
#   might need to actually blacklist them or write out a doc db
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
LOAD_VOCAB = False                                       # whether to build vocab (slow) or read from disk
LOAD_FEATURES = False                                    # whether to compute features (slow) or read from disk

PDF_ROOT = 'allpdfs'                                    # where to search for PDFs
STOP_FILE = 'stoplist.txt'                              # http://jmlr.csail.mit.edu/papers/volume5/lewis04a/a11-smart-stop-list/english.stop
DOCS_FILE = PDF_ROOT + '_db.txt'                        # flat db of dumpable pdfs -- need to save these to keep indexing into features correct
VOCAB_FILE = PDF_ROOT + '_vocab.json'                   # vocab file to read from/write to disk
FEATURES_FILE_STEM = PDF_ROOT + '_features'             # features file stem name
FEATURES_FILE_LOAD = PDF_ROOT + '_features_tf_idf.mtx'  # actual features to load
GRAPH_FILE = PDF_ROOT + '_graph.net'                    # graph file to write
N_REDUCE_FEATURES = 100                                 # dimensionality of reduced features
WEIGHT_THRESH = 0.75                                    # thresh for writing out an edge in threshold output function
KNN_K = 5                                               # how many edges (k) should we write out in knn output function

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
    nlpfuns.WriteGraphPajekKNN(GRAPH_FILE, [x.split(os.sep)[-1] for x in docs], 1-distances, KNN_K)

    print 'Done. (elapsed time %f secs)' % (time.time() - starttime)