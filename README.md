# Build

A `jar` file for a "stable" version is accompanied in folder `dist`. To build the latest code:

- `ant jar`: to build jar file 
- `ant clean-build`: to make a clean build

Take a look at the `build.xml` for more options.

# LDA
```
java -cp "$SEGAN_PATH/dist/segan.jar:$SEGAN_PATH/lib/*" sampler.unsupervised.LDA --dataset <dataset-name> --word-voc-file <word-vocab-file> --word-file <doc-word-file> --info-file <doc-info-file> --output-folder <output-folder> --burnIn <number-burn-in-iterations> --maxIter <max-number-iterations> --sampleLag <sample-lag> --report <report-interval> --K <number-topics> --alpha <alpha> --beta <beta>
```

# HDP
```
java -cp "$SEGAN_PATH/dist/segan.jar:$SEGAN_PATH/lib/*" sampler.unsupervised.HDP --dataset <dataset-name> --word-voc-file <word-vocab-file> --word-file <doc-word-file> --info-file <doc-info-file> --output-folder <output-folder> --burnIn <number-burn-in-iterations> --maxIter <max-number-iterations> --sampleLag <sample-lag> --report <report-interval> --K <initial-num-topics> --global-alpha <global-alpha> --local-alpha <local-alpha> --beta <beta> --init <initialization>
```

# SLDA
This provides an implementation of Supervised Latent Dirichlet allocation (Blei and McAuliffe, NIPS'07). SLDA's input is a set of documents, each of which is associated with a continuous response variable.

### Running SLDA
```
java -cp "$SEGAN_PATH/dist/segan.jar:$SEGAN_PATH/lib/*" sampler.supervised.regression.SLDA --dataset <dataset-name> --word-voc-file <word-vocab-file> --word-file <doc-word-file> --info-file <doc-info-file> --output-folder <output-folder> --burnIn <number-burn-in-iterations> --maxIter <max-number-iterations> --sampleLag <sample-lag> --report <report-interval> --K <number-topics> --alpha <alpha> --beta <beta> --rho <rho> --sigma <sigma> --mu <mu> --init <initialization>
```

- `<dataset-name>`: Name of the dataset
- `<word-vocab-file>`: File contains the word vocabulary. Each word type is separated by a new line.
- `<doc-word-file>`: File contains the formatted text of documents, which has the following format:
```
<num-word-types-doc-1> <word-type-1>:<frequency> <word-type-2>:<frequency> ...\n
<num-word-types-doc-2> <word-type-1>:<frequency> <word-type-2>:<frequency> ...\n
...
<num-word-types-doc-D> <word-type-1>:<frequency> <word-type-2>:<frequency> ...\n
```
In `<doc-word-file>`, each line represents a document. The first number (e.g., `<num-word-types-doc-d>`) is the number of unique word types in the document. Subsequently, each pair `<word-type-1>:<frequency>` represents a word type and its frequency in the document.

- `<doc-info-file>`: File contains the document IDs and their associated responses, which has the following format:
```
<doc-1-ID>\t<response-1>\n
<doc-2-ID>\t<response-2>\n
...
<doc-D-ID>\t<response-D>\n
```
- `<output-folder>`: Folder to output results
- `<number-burn-in-iterations>`: Number of burn-in iterations (default: 500)
- `<max-number-iterations>`: Maximum number of iterations (default: 1000)
- `<sample-lag>`: Sample lag (default: 50)
- `<report-interval>`: Number of iterations between printing verbosely to console (default: 25)
- `<number-topics>`: Number of topics (default: 50)
- `<alpha>`: Document-topic Dirichlet hyperparameter (default: 0.1)
- `<beta>`: Topic-word Dirichlet hyperparameter (default: 0.1)
- `<rho>`: Variance of the Gaussian distribution for documents' responses (default: 1.0)
- `<sigma>`: Variance of the Gaussian prior for topics' regression parameters (default: 1.0)
- `<mu>`: Mean of the Gaussian prior for topics' regression parameters (default: 0.0)
- `<initialization>`: Initialization of SLDA. There are current two options: (1) `random`: randomly assign topic to token (default) and (2) `preset`: run LDA for 100 iterations and use LDA's assignments for SLDA's initialization.
- `-z`: Whether the documents' responses are z-normalized
- `-v`: Verbose
- `-d`: Debug

### Processing data for SLDA
You can either preprocess your raw data into the format described above to run SLDA or use the existing tool provided in `segan`. To process data using `segan`:
```
java -cp 'dist/segan.jar:dist/lib/*' data.ResponseTextDataset --dataset <dataset-name> --text-data <input-text> --data-folder <data-folder> --format-folder <format-folder> --run-mode process --response-file <response-file>
```
- `<dataset-name>`:	The name of the dataset
- `<input-text>`:	Raw input documents. This can be either (1) a file in which each line is a document, or (2) a folder in which each file is a document. If `<input-text>` is a file, each line has to have the format `<doc_id>\t<doc_content>\n`. If `<input-text>` is a folder, each file stores the document text and the file name will be treated as the `doc_id`.
- `<data-folder>`:	The folder where the processed data will be stored
- `<format-folder>`:	The subfolder that the processed data will be stored. More specifically, after running this, the processed data will be stored in `<output-folder>/<dataset-name>/<format-folder>`.
- `<format-file>`(optional): By default, all processed data files will be named `<dataset-name>.<extension>`, e.g., `<dataset-name>.dat`, `<dataset-name>.wvoc` etc. If you want to rename these formatted files to `<format-file>.dat`, `<format-file>.wvoc` etc, use this option.
- `<response-file>`: File contains the documents' responses
Other important input arguments:
- `--u`: Minimum unigram frequency (default: 1)
- `--b`: Minimum bigram frequency (default: 1)
- `--bs`: Minimum bigram chi-squared score (default: 5)
- `--V`: Maximum vocabulary size
- `-s`: Whether stop words are filtered
- `-l`: Whether stemming/lemmatization is performed
- `-v`: Verbose
- `-d`: Debug 

### Running example
```
java -cp 'dist/segan.jar:lib/*' data.ResponseTextDataset --dataset amazon-data --text-data demo/amazon-data/raw/text.txt --response-file demo/amazon-data/raw/response.txt --data-folder demo --format-folder format-supervised --run-mode process -v -d --u 5 -s -l --bs 10 --b 5 --V 10000
```

```
java -cp "dist/segan.jar:lib/*" sampler.supervised.regression.SLDA --dataset amazon-data --word-voc-file demo/amazon-data/format-supervised/amazon-data.wvoc --word-file demo/amazon-data/format-supervised/amazon-data.dat --info-file demo/amazon-data/format-supervised/amazon-data.docinfo --output-folder demo/amazon-data/supervised-models --burnIn 250 --maxIter 500 --sampleLag 25 --report 5 --K 50 --alpha 0.1 --beta 0.1 --rho 1.0 --sigma 1.0 --mu 0.0 --init random -v -d -z
```

Notes:
- To avoid the Java Heap Space Errors, increase `-Xmx` and `-Xms`. For example, `-Xmx10000M -Xms10000M`.
- For other input arguments, use `-help`. For example, `java -cp dist/segan.jar data.ResponseTextData -help`.
- For each document in the input folder, the filename is used as its ID. If the filename has `txt` as the file extension, `txt` will be discarded. For example, a document stored in file `doc-1.txt` will have `doc-1` as its ID, whereas a document stored in file `doc-2.dat` will have `doc-2.dat` as its ID.

# Binary SLDA
SLDA for a set of documents, each of which is associated with a binary response variable (0 or 1). 
### Running BinarySLDA
```
java -cp "$SEGAN_PATH/dist/segan.jar:$SEGAN_PATH/lib/*" sampler.supervised.classification.BinarySLDA --dataset <dataset-name> --word-voc-file <word-vocab-file> --word-file <doc-word-file> --info-file <doc-info-file> --output-folder <output-folder>
```
- `<dataset-name>`:	The name of the dataset
- `<output-folder>`: Folder to output results
- `<word-vocab-file>`: File contains the word vocabulary. Each word type is separated by a new line.
- `<doc-word-file>`: File contains the formatted text of documents (see above).
- `<doc-info-file>`: File contains the document IDs and their associated binary label.
```
<doc-1-ID>\t<label-1>\n
<doc-2-ID>\t<label-2>\n
...
<doc-D-ID>\t<label-D>\n
```
In addition, here are optional inputs and their default values:
- `--K`: Number of topics (default: 50)
- `--burnIn`: Number of burn-in iterations (default: 500)
- `--maxIter`: Maximum number of iterations (default: 1000)
- `--sampleLag`: Sample lag (default: 50)
- `--report`: Number of iterations between printing verbosely to console (default: 25)
- `--alpha`: Document-topic Dirichlet hyperparameter (default: 0.1)
- `--beta`: Topic-word Dirichlet hyperparameter (default: 0.1)
- `--sigma`: Variance of the Gaussian prior for topics' regression parameters (default: 1.0)
- `--mu`: Mean of the Gaussian prior for topics' regression parameters (default: 0.0)
- `--init`: Initialization of SLDA. There are currently two options: 
 - `random`: randomly assign topic to token (default) 
 - `preset`: run LDA for 100 iterations and use LDA's assignments for SLDA's initialization.
- `-v`: Verbose
- `-d`: Debug

### Processing data for BinarySLDA
Example of input data for BinarySLDA is provided in folder `demo/amazon-data/format-binary`. You can either process your raw data into the same format as provided in `demo/amazon-data/format-binary`, or use `segan`'s processing tool.
```
java -cp "$SEGAN_PATH/dist/segan.jar:$SEGAN_PATH/lib/*" data.LabelTextDataset --dataset <dataset-name> --text-data <input-text> --data-folder <data-folder> --format-folder <format-folder> --run-mode process --label-file <label-file>
```
- `<dataset-name>`:	The name of the dataset
- `<input-text>`:	Raw input documents. This can be either (1) a file in which each line is a document, or (2) a folder in which each file is a document. If `<input-text>` is a file, each line has to have the format `<doc_id>\t<doc_content>\n`. If `<input-text>` is a folder, each file stores the document text and the file name will be treated as the `doc_id`.
- `<data-folder>`:	The folder where the processed data will be stored
- `<format-folder>`:	The subfolder that the processed data will be stored. More specifically, after running this, the processed data will be stored in `<output-folder>/<dataset-name>/<format-folder>`.
- `<format-file>`(optional): By default, all processed data files will be named `<dataset-name>.<extension>`, e.g., `<dataset-name>.dat`, `<dataset-name>.wvoc` etc. If you want to rename these formatted files to `<format-file>.dat`, `<format-file>.wvoc` etc, use this option.
- `<label-file>`: File contains the documents' binary label
Other important input arguments:
- `--u`: Minimum unigram frequency (default: 1)
- `--b`: Minimum bigram frequency (default: 1)
- `--bs`: Minimum bigram chi-squared score (default: 5)
- `--V`: Maximum vocabulary size
- `-s`: Whether stop words are filtered
- `-l`: Whether stemming/lemmatization is performed
- `-v`: Verbose
- `-d`: Debug 

### Running example
- To process raw data 
```
java -cp 'dist/segan.jar:dist/lib/*' data.LabelTextDataset --dataset amazon-data --text-data demo/amazon-data/raw/text.txt --data-folder demo --format-folder format-binary --run-mode process --label-file demo/amazon-data/raw/binary-label.txt -v -d --u 5 -s -l --bs 10 --b 5 --V 10000
```
- To run BinarySLDA on formatted data
```
java -cp "dist/segan.jar:lib/*" sampler.supervised.classification.BinarySLDA --dataset amazon-data --word-voc-file demo/amazon-data/format-binary/amazon-data.wvoc --word-file demo/amazon-data/format-binary/amazon-data.dat --info-file demo/amazon-data/format-binary/amazon-data.docinfo --output-folder demo/amazon-data/binary-supervised-models --burnIn 100 --maxIter 250 --sampleLag 30 --report 5 --K 50 --alpha 0.1 --beta 0.1 --sigma 1.0 --mu 0.0 --init random -v -d
```

# SHDP
```
 java -cp "$SEGAN_PATH/dist/segan.jar:$SEGAN_PATH/lib/*" sampler.supervised.regression.SHDP --dataset <dataset-name> --word-voc-file <word-vocab-file> --word-file <doc-word-file> --info-file <doc-info-file> --output-folder <output-folder> --burnIn <number-burn-in-iterations> --maxIter <max-number-iterations> --sampleLag <sample-lag> --report <report-interval> --global-alpha <global-alpha> --local-alpha <local-alpha> --beta <beta> --rho <rho> --sigma <sigma> --mu <mu> --init <initialization>
```
