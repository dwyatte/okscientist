import gensim.corpora as corpora
import gensim.models as models
import gensim.similarities as similarities
import scipy.sparse as sparse
import numpy
import sys, os, time, logging
import nlpfuns

PDF_ROOT = 'allpdfs'                                    # where to search for PDFs
STOP_FILE = 'stoplist.txt'                              # http://jmlr.csail.mit.edu/papers/volume5/lewis04a/a11-smart-stop-list/english.stop
GRAPH_STEM = PDF_ROOT + '_gensim_lda_graph'             # stem of graph file to write
NUM_TOPICS = 100                                        # dimensionality for lsi/lda model
WEIGHT_THRESH = 0.75                                    # threshold for including an edge in threshold graph function
KNN_K = 5                                               # how many edges (k) should we include out per node in knn graph function

if __name__ == '__main__':

    starttime = time.time()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # if we are on a Mac, we can use the supplied pdftotext binary
    # TODO: make sure it is in our path, else error
    if sys.platform == 'darwin':
        os.environ['PATH'] = 'bin:' + os.environ['PATH']                      

    print '\nBuilding Dictionary (could take awhile)...'
    docs = nlpfuns.FindPDFs(PDF_ROOT)
    stoplist = nlpfuns.ReadFlatText(STOP_FILE)
    docsrm = []    
    dictionary = corpora.Dictionary() # a mapping of words to word-ids that gets updated for each new doc
    corpus = [] # bag of words representation using the dictionary for each doc
    
    for doc,didx in zip(docs, range(len(docs))):
        print 'Adding text from %s (%d/%d)' % (doc, didx+1, len(docs))
        doctext = nlpfuns.DumpPDF(doc, filterjunk=True, stoplist=stoplist)
        # if doc was not dumpable (i.e., image-based pdf, or copyrighted, etc.), rm from list
        if len(doctext) > 0:
            # make bag of words and update dictionary, add to corpus
            bow = dictionary.doc2bow(doctext, allow_update=True)
            corpus.append(bow)
        else:
            print '|---> Could not dump text from this paper, scheduling for removal\n'
            docsrm.append(doc)
            continue
    print '\nRemoving %d docs from db' % (len(docsrm))        
    docs = [x for x in docs if x not in docsrm]
    print 'Done.'
            
    # extract tf-idf features from corpus object and create lsi model
    print '\nBuilding models...'
    # tf-idf model
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]        
    # # lsi model
    # lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=NUM_TOPICS)
    # corpus_tfidf_lsi = lsi[corpus_tfidf]
    # corpus_tfidf_lsi_index = similarities.SparseMatrixSimilarity(corpus_tfidf_lsi, num_features=NUM_TOPICS)
    # lda model
    lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=NUM_TOPICS)
    corpus_tfidf_lda = lda[corpus_tfidf]
    corpus_tfidf_lda_index = similarities.SparseMatrixSimilarity(corpus_tfidf_lda, num_features=NUM_TOPICS)
    
    print 'Creating graphs and writing to Pajek file...'
    # similarities -- just do them all in memory, but gensim author notes that this won't 
    # scale for very large document libraries (1mil+)

    similarities = numpy.zeros((len(docs), len(docs)))
    for similarity,sidx in zip(corpus_tfidf_lda_index, range(len(corpus_tfidf_lda_index))):
        similarities[sidx,:] = numpy.array(similarity)

    graphthresh = nlpfuns.CreateGraphThresh(docs, similarities, WEIGHT_THRESH)
    graphthresh = nlpfuns.ReduceGraphUndirected(graphthresh)
    # list comprehension just strips off last part of file path for node label
    nlpfuns.WriteGraphPajek(GRAPH_STEM+'_thresh'+str(WEIGHT_THRESH)+'.net', graphthresh, [x.split(os.sep)[-1] for x in docs])
    
    graphknn = nlpfuns.CreateGraphKNN(docs, similarities, KNN_K)
    graphknn = nlpfuns.ReduceGraphUndirected(graphknn)
    # list comprehension just strips off last part of file path for node label
    nlpfuns.WriteGraphPajek(GRAPH_STEM+'_knn'+str(KNN_K)+'.net', graphknn, [x.split(os.sep)[-1] for x in docs])
    
    print 'Done. (elapsed time %f secs)\n' % (time.time() - starttime)