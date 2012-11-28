import nlpfuns
import sys, os
import numpy
import scipy.spatial.distance as distance


# various flags/parameters
DOC_ROOT = 'allpdfs/'                          # where to search for PDFs
BUILD_VOCAB = False                             # whether to build vocab (slow) or read from disk
VOCAB_FILE = 'vocab.json'                       # vocab file to read from/write to disk
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
        print '\nReading existing vocabulary from %s\n' % (VOCAB_FILE)
        vocab = nlpfuns.ReadJSON(VOCAB_FILE)
 
    
    # Compute features, which are a dict (can also be saved in json format). The keys are
    # the PDF filenames, which we just strip off the paths. The feature values should be
    # the same size as the vocab values
    features = {} 
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

        docname = doc.split(os.sep)[-1]
        features[docname] = nlpfuns.ComputeFreqFeatures(doctext, vocab)
    
    # convert features to tf-idf and compute distance
    print '\nComputing TF-IDF features and inter-document distance...'
    features = nlpfuns.ComputeTFIDFFeatures(features, vocab)            	    
    distances = distance.pdist(features.values(), 'cosine')
    distances = distance.squareform(distances)
    
    print 'Writing Pajek graph file...'
    nlpfuns.WriteGraphPajek('graph.net', features.keys(), 1-distances, WEIGHT_THRESH)
    print 'Done'