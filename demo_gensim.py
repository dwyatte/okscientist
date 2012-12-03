import gensim.corpora as corpora
import gensim.models as models
import nlpfuns
import sys, os, time, logging

PDF_ROOT = 'allpdfs'                                    # where to search for PDFs
STOP_FILE = 'stoplist.txt'                              # http://jmlr.csail.mit.edu/papers/volume5/lewis04a/a11-smart-stop-list/english.stop
DOCS_FILE = PDF_ROOT + '_db.txt'                        # flat db of dumpable pdfs -- need to save these to keep indexing into features corr
NUM_TOPICS = 100

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
    dictionary = corpora.Dictionary() 
    corpus = []
    
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
    print '\nDone.'
            
    # extract tf-idf features from corpus object and create lsi model
    print '\nBuilding models...'
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    # create the lsi model
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=NUM_TOPICS)
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=NUM_TOPICS)