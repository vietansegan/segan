# Compile

    cd <SEGAN_PATH>
    ant jar
    cp -r lib dist/

# Process Data

The main kind of data that `segan` deals with is textual data in which there is a collection of documents, each of which is associated with some additional values which can be either continuous (response variable) or discrete (label) or both. The texts can be stored in either a big file (each line is a document) or a folder (each file is a document).

## Text-only data

### Processing multiple files in a folder

To process text-only data stored in multiple files in a folder, each file corresponds to a document, use the following command:

    java -cp 'dist/segan.jar:dist/lib/*' data.TextDataset --dataset <dataset-name> --text-data <input-text-folder> --data-folder <data-folder> --format-folder <format-folder> --run-mode process
    
 - `<dataset-name>`:	The name of the dataset
 - `<input-text-folder>`:	The folder that contains input text data, in which each file is one document
 - `<data-folder>`:	The folder where the processed data will be stored
 - `<format-folder>`:	The subfolder that the processed data will be stored. More specifically, after running this, the processed data will be stored in `<output-folder>/<dataset-name>/<format-folder>`.
 - `<format-file>`(optional): By default, all processed data files will be named `<dataset-name>.<extension>`, e.g., `<dataset-name>.dat`, `<dataset-name>.wvoc` etc. If you want to rename the formatted file, use this option.

#### Notes

 - For other arguments, use `-help`. For example: `java -cp dist/segan.jar data.TextDataset -help`
 - For each document in the input folder, the filename is used as its ID. If the filename has `txt` as the file extension, `txt` will be discarded. For example, a document stored in file `doc-1.txt` will have `doc-1` as its ID, whereas a document stored in file `doc-2.dat` will have `doc-2.dat` as its ID.

### Processing single file

To process text-only data stored in a single file, where each line corresponds to a document with the following format `<doc_id>\t<doc_content>\n`, use the foloowing command:

    java -cp 'dist/segan.jar:dist/lib/*' data.TextDataset --dataset <dataset-name> -file --text-data <input-text-file> --data-folder <data-folder> --format-folder <format-folder> --run-mode process
    
 - `<input-text-file>`:	The file that contains input text data, in which each line is one document.

## Text data with continuous response variable

This is to process a collection of documents, each of which is associated with a single continuous response variable. This is done similarly as above, but using `data.ResponseTextDataset` with an additional argument `<response-file>` which contains the value of the response variable. The `<response-file>` has the following format: `<docid>\t<response_value>\n`.

    java -cp 'dist/segan.jar:dist/lib/*' data.ResponseTextDataset --dataset <dataset-name> --text-data <input-text-file> --data-folder <data-folder> --format-folder <format-folder> --run-mode process --response-file <response-file>

    java -cp 'dist/segan.jar:dist/lib/*' data.ResponseTextDataset --dataset <dataset-name> -file --text-data <input-text-folder> --data-folder <data-folder> --format-folder <format-folder> --run-mode process --response-file <response-file>
    
## Text data with categorical response variable

Under construction.
    
## Create cross validation data

Currently only support data with continuous responses

    java -cp 'dist/segan.jar:dist/lib/*' data.ResponseTextDataset--dataset <dataset-name> --text-data <input-text-folder> --response-file <response_value> --data-folder <data-folder> --cv-folder <cross-validation-folder> --num-folds <number-of-folds> --tr2dev-ratio <training-to-development-ratio> --num-classes <number-of-discretized-classes> --run-mode cross-validation

 - `<cv-folder>`: Directory of the folder to contain the processed cross-validation data.
 - `<num-folds>`: The number of folds
 - `<tr2dev-ratio>`: Training-to-development ratio: the ratio between the number of training instances to the number of development instances.
 - `<num-classes>` is the number of classes that the response variable is discretized into, which is used for stratified sampling. For example, if `<num-classes> = 3`, the response variables are discretized into 3 classes, and CreateCrossValidationFolds will try to preserve the distributions of these 3 classes in training, development and test sets. Default, `<num-classes> = 1`.

# Run SLDA (to be revised)

Run SLDA Gibbs sampler with the default setting:
    
    java -cp 'dist/segan.jar:dist/lib/*:/fs/clip-ml/gurobi502/linux64/lib/gurobi.jar' sampler.supervised.regression.SLDA --dataset <dataset-name> --data-folder <data-folder> --format-folder <format-folder> --output <result-folder> --K <number-of-topics> -z
   
   E.g.,
   java -Xmx4096M -Xms4096M -cp 'dist/segan.jar:dist/lib/*:/fs/clip-ml/gurobi502/linux64/lib/gurobi.jar' sampler.supervised.regression.SLDA --dataset amazon-data --data-folder demo --format-folder format-response --output demo/amazon-data/format-response-model --K 50 --burnIn 10 --maxIter 20 --sampleLag 5 -v -d -z

2. Run SLDA Gibbs sampler for cross validation
   a) To train on training data
   
   java -cp 'dist/segan.jar:dist/lib/*:/fs/clip-ml/gurobi502/linux64/lib/gurobi.jar' sampler.supervised.regression.SLDA --dataset <dataset-name> --cv-folder <cross-validation-folder> --num-folds <number-of-folds> --output <result-folder> --K <number-of-topics> --run-mode train-test -v -d -z

   b) To test, use "--run-mode test" instead of "--run-mode train"

   c) To do both training and test, use "--run-mode train-test" instead
   
   E.g.,
   java -Xmx4096M -Xms4096M -cp 'dist/segan.jar:dist/lib/*:/fs/clip-ml/gurobi502/linux64/lib/gurobi.jar' sampler.supervised.regression.SLDA --dataset amazon-data --cv-folder demo/amazon-data/format-cv-5-0.8 --num-folds 5 --output demo/amazon-data/format-cv-5-0.8-model --K 50 --burnIn 10 --maxIter 20 --sampleLag 5 --fold 0 --run-mode train-test -v -d -z

   *** Notes:
   + --fold: to specify a single fold that you want to run on
   + -z: to perform z-normalization on the response variable
   + -v: verbose
   + -d: debug

# Run SHLDA

## Train

    java  -cp 'dist/segan.jar:dist/lib/*:/fs/clip-ml/gurobi502/linux64/lib/gurobi.jar' sampler.supervised.regression.SHLDA --dataset <dataset-name> --data-folder <data-folder> --format-folder <format-folder> --output <result-folder> --run-mode train 

 - `<dataset>`, `<data-folder>`, `<format-folder>`: See above
 - `<output>`: Directory of the output folder
 - `<tree-height>`(optional): The height of the tree. Default: `3`. 
 - `<init-branch-factor>`(optional): An array to specify the branching factors of the initial tree. For example, use `10-5-3` to initialize a tree having 10 first-level nodes, each of which has 5 second-level children, each of which in turn has 3 third-level children. Default: `15-3`. 
 - `<seeded-asgn-file>`(optional): File containing a pre-learned topic assignments for each tokens. This is to initialize SHLDA with good first-level topics (for example, topics learned from iteractive topic models).
 - `<T>`(optional): The parameter for L1-norm regularization on lexical regression. The smaller this number is, the fewer lexical items are used (in other words, there'll be more lexical items haiving zero weights). Default: `500`.

#### Notes
 - For other arguments, use `-help`: `java -cp 'dist/segan.jar' sampler.supervised.regression.SHLDA -help`
 - Use `-Xmx` and `-Xms` to specify Java heap size. For example `-Xmx4096M -Xms4096M`.
    
## Test

    java -cp 'dist/segan.jar:dist/lib/*:/fs/clip-ml/gurobi502/linux64/lib/gurobi.jar' sampler.supervised.regression.SHLDA --dataset <dataset-name> --data-folder <data-folder> --format-folder <format-folder> --output <result-folder> --run-mode test
    
## Cross validation

    java -cp 'dist/segan.jar:dist/lib/*:/fs/clip-ml/gurobi502/linux64/lib/gurobi.jar' sampler.supervised.regression.SHLDA --dataset <dataset-name> --output <result-folder> --cv-folder <cross-validation-folder> --num-folds <number-of-folds> --fold <fold-number> --run-mode <running-mode>
    
 - `<cv-folder>`: The cross-validation folder (see above).
 - `<num-folds>`: Number of folds
 - `<fold-number>`(optional): The fold number to run. If this argument is not set, all folds will be run.
 - `<run-mode>`(optional): The running mode which can be either `train` (run on training data only), `test` (evaluate on test data only) or `train-test` (train on training data and evaluate on test data). Default: `train-test`.
