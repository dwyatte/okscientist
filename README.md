okscienctist
====
okscientist was a project for an informatics class that computed similarities between libraries of documents.
The ultimate goal of the project was to infer compatibilities between users in terms of their libraries of PDF documents
in order to identify potential collaborators (think okcupid for scientists, hence "okscientist"). As a cursory goal, 
it simply visualizes the similarity structure of a library of PDF documents. In theory, the code could be adopted to work 
with the [Mendeley API](http://dev.mendeley.com/) (or any number of similar services).

Installation
----
1.) Install Scipy/Numpy if you don't already have it (requires Xcode to be installed on Mac OS).
The easiest method is to use the install_superpack.sh script included in the deps directory.  
cd deps  
./install_superpack.sh  
cd ..  

2.) Install gensim. The easiest way to do this is with python's easy_install tool. If you are
using Mac OS 10.8.X, there is a known bug with the Scipy/Numpy (as of 12/2012) where dependency
checking doesn't work quite properly. The easiest way is to run the install_3rdparty.sh script
included in the deps directory.  
cd deps  
./install_3rdparty.sh  
cd ..  

3.) Install Xpdf/pdftotext. If you are using Mac OS, we have included a binary in the bin
directory that will automatically get added to your path. If the binary is included somewhere
nonstandard, make sure to manually add it to your path.

Usage:
----
1.) Create a symbolic link to your PDF library called 'allpdfs' in the okscientist directory
(the code looks in this directory by default, would be better to pass as an argument).

2.) Run the code using 'python demo.py'. This will build the vocabulary and all features, 
which will take awhile (around 30 minutes for my library of 1000 PDFs). Fortunately, the code 
saves all of these things to disk, which can then be loaded on subsequent runs by setting the
flags LOAD_VOCAB and LOAD_FEATURES to True (again, would be better to pass these as arguments).

The code is fairly optimized for memory, and should scale to fairly large libraries. gensim
includes even better memory optimization as well as online-updating LSI and LDA models (opposed 
to the implementation here which makes two separate passes for calculating the overall vocabulary
and then the features) -- see below for more information.

3.) Optional: There is also a demo that uses gensim for the LSI and LDA (default) computations
instead of the implementation in nlpfuns.py. Run 'python demo_gensim.py' for this version.

Analysis:
----
The code writes out undirected Pajek formatted graph files (.net extension), which can be 
analyzed using a variety of software packages, e.g., gephi (https://gephi.org/).
