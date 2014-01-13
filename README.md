# Compile

    cd <SEGAN_PATH>
    ant jar
    cp -r lib dist/

# Process Data

The main kind of data that `segan` deals with is textual data in which there is a collection of documents, each of which is associated with some additional values which can be either continuous (response variable) or discrete (label) or both. The texts can be stored in either a big file (each line is a document) or a folder (each file is a document).

## Process text-only data

Processing text-only data from multiple files from a folder. The filename will be the document id.

    java -cp 'dist/segan.jar:dist/lib/*' data.TextDataset --dataset <dataset-name> --text-data <input-text-folder> --data-folder <data-folder> --format-folder <format-folder> --run-mode process

   - `<dataset-name>`:	The name of the dataset
   - `<input-text-folder>`:	The folder that contains input text data, in which each file is one document
   - `<data-folder>`:	The folder where the processed data will be stored
   - `<format-folder>`:	The subfolder that the processed data will be stored. More specifically, after running this, the processed data will be stored in `<output-folder>/<dataset-name>/<format-folder>`.

For other options, use `-help`. For example

    java -cp dist/segan.jar data.TextDataset -help

Processing text-only data from a single file, in which each line is a document with the following format: `<doc_id>\t<doc_content>\n`:

    java -cp 'dist/segan.jar:dist/lib/*' data.TextDataset --dataset <dataset-name> -file --text-data <input-text-folder> --data-folder <data-folder> --format-folder <format-folder> --run-mode process

   Other options:
   - `<format-file>`: By default, all processed data files will be named `<dataset-name>.<extension>`, e.g., `<dataset-name>.dat`, `<dataset-name>.wvoc` etc. If you want to rename the formatted file, use this option.

## Process textual data with continuous response variable

Processing data in which each document is associated with a single continuous value. This can be done similarly as above with an additional argument `<response-file>` which contains the value of the response variable. The `<response-file>` has the following format: `<docid>\t<response_value>\n`

## Create cross validation data (currently only support data with continuous responses)

    java -cp 'dist/segan.jar:dist/lib/*' data.ResponseTextDataset--dataset <dataset-name> --text-data <input-text-folder> --response-file <response_value> --data-folder <data-folder> --cv-folder <cross-validation-folder> --num-folds <number-of-folds> --tr2dev-ratio <training-to-development-ratio> --num-classes <number-of-discretized-classes> --run-mode cross-validation
   
   *** Note: `<num-classes>` is the number of classes that the response variable is discretized into, which is used for stratified sampling. For example, if `<num-classes> = 3`, the response variables are discretized into 3 classes, and CreateCrossValidationFolds will try to preserve the distributions of these 3 classes in training, development and test sets. Default, `<num-classes> = 1`.

Run SLDA
--------
1. Run SLDA Gibbs sampler with the default setting
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

Run SHLDA
---------
1. Run SHLDA
   java -Xmx4096M -Xms4096M -cp 'dist/segan.jar:dist/lib/*:/fs/clip-ml/gurobi502/linux64/lib/gurobi.jar' sampler.supervised.regression.SHLDA --dataset <dataset-name> --data-folder <data-folder> --format-folder <format-folder> --output <result-folder> --num-topics <initial number of 1st level nodes> --num-frames <initial number of 2nd level nodes per 1st node> --T <L1-norm regularizer param>
   
   Notes:
   --num-topics: number of 1st-level nodes for initialization
   --num-frames: number of 2nd-level nodes per 1st-level node for initialization. Currently only this 2-level structure is suppported for initialization. Will have arbitrary number of levels in the future.
   --T: the parameter for L1-norm regularization on lexical regression. The smaller this number is, the few lexical items are used (in other words, there'll be more lexical items having zero weights).
   
   E.g.,
   java -Xmx4096M -Xms4096M -cp 'dist/segan.jar:dist/lib/*:/fs/clip-ml/gurobi502/linux64/lib/gurobi.jar' sampler.supervised.regression.SHLDA --dataset amazon-data --data-folder demo --format-folder format-response --output demo/amazon-data/format-response-model --burnIn 10 --maxIter 20 --sampleLag 5 -v -d -z --num-topics 15 --num-frames 3 --T 500
   
2. Run SHLDA for cross validation
   java -Xmx4096M -Xms4096M -cp 'dist/segan.jar:dist/lib/*:/fs/clip-ml/gurobi502/linux64/lib/gurobi.jar' sampler.supervised.regression.SHLDA --dataset <dataset-name> <cross-validation-folder> --num-folds <number-of-folds> --output <result-folder> --num-topics <initial number of 1st level nodes> --num-frames <initial number of 2nd level nodes per 1st node> --T <L1-norm regularizer param> --run-mode train-test -z 

   E.g.,
   java -Xmx4096M -Xms4096M -cp 'dist/segan.jar:dist/lib/*:/fs/clip-ml/gurobi502/linux64/lib/gurobi.jar' sampler.supervised.regression.SHLDA --dataset amazon-data --cv-folder demo/amazon-data/format-cv-5-0.8 --num-folds 5 --output demo/amazon-data/format-cv-5-0.8-model --burnIn 10 --maxIter 20 --sampleLag 5 --fold 0 --run-mode train-test -v -d -z --num-topics 15 --num-frames 3 --T 500
   
=== Examples to process and run models with the accompanied amazon data ===
* Process data
java -Xmx4096M -Xms4096M-cp 'dist/segan.jar:dist/lib/*' data.ResponseTextDataset --dataset amazon-data -file --text-data demo/amazon-data/raw/text.txt --response-file demo/amazon-data/raw/response.txt --data-folder demo --format-folder format-response --u 5 --b 3 --bs 10 -s -l --V 5000 --run-mode process -v

* Create cross validation data
java -Xmx4096M -Xms4096M-cp 'dist/segan.jar:dist/lib/*' data.ResponseTextDataset --dataset amazon-data -file --text-data demo/amazon-data/raw/text.txt --response-file demo/amazon-data/raw/response.txt --data-folder demo --cv-folder demo/amazon-data/format-cv-5-0.8 --num-folds 5 --tr2dev-ratio 0.8 --num-classes 5 --u 5 --b 3 --bs 10 -s -l --V 5000 --run-mode cross-validation -v

* Run SLDA on the processed data
java -Xmx4096M -Xms4096M -cp 'dist/segan.jar:dist/lib/*:/fs/clip-ml/gurobi502/linux64/lib/gurobi.jar' sampler.supervised.regression.SLDA --dataset amazon-data --data-folder demo --format-folder format-response --output demo/amazon-data/format-response-model --K 50 --burnIn 10 --maxIter 20 --sampleLag 5 -v -d -z

* Run SLDA on the first fold of the cross-validated data
java -Xmx4096M -Xms4096M -cp 'dist/segan.jar:dist/lib/*:/fs/clip-ml/gurobi502/linux64/lib/gurobi.jar' sampler.supervised.regression.SLDA --dataset amazon-data --cv-folder demo/amazon-data/format-cv-5-0.8 --num-folds 5 --output demo/amazon-data/format-cv-5-0.8-model --K 50 --burnIn 10 --maxIter 20 --sampleLag 5 --fold 0 --run-mode train-test -v -d -z

* Run SHLDA on the processed data
java -Xmx4096M -Xms4096M -cp 'dist/segan.jar:dist/lib/*:/fs/clip-ml/gurobi502/linux64/lib/gurobi.jar' sampler.supervised.regression.SHLDA --dataset amazon-data --data-folder demo --format-folder format-response --output demo/amazon-data/format-response-model --burnIn 10 --maxIter 20 --sampleLag 5 -v -d -z --num-topics 15 --num-frames 3 --T 500

* Run SHLDA on the first fold of the cross-validated data
java -Xmx4096M -Xms4096M -cp 'dist/segan.jar:dist/lib/*:/fs/clip-ml/gurobi502/linux64/lib/gurobi.jar' sampler.supervised.regression.SHLDA --dataset amazon-data --cv-folder demo/amazon-data/format-cv-5-0.8 --num-folds 5 --output demo/amazon-data/format-cv-5-0.8-model --burnIn 10 --maxIter 20 --sampleLag 5 --fold 0 --run-mode train-test -v -d -z --num-topics 15 --num-frames 3 --T 500 
