=== COMPILE ===
    cd <SEGAN_PATH>
    ant jar
    cp -r lib dist/

=== PROCESS DATA ===
The main kind of data that segan deals with is textual data in which there is a collection of documents, each of which is associated with some additional values which can be either continuous (response variable) or discrete (label) or both.

The texts can be stored in either a big file (each line is a document) or a folder (each file is a document).

1. Processing text-only data from multiple files from a folder. The filename will be the document id.
   java -cp 'dist/segan.jar:dist/lib/*' main.ProcessData --dataset <dataset-name> --text-data <input-text-folder> --data-folder <data-folder> --format-folder <format-folder>

   - <dataset-name>:	The name of the dataset
   - <input-text-folder>:	The folder that contains input text data, in which each file is one document
   - <data-folder>:	The folder where the processed data will be stored
   - <format-folder>:	The subfolder that the processed data will be stored. More specifically, after running this, the processed data will be stored in "<output-folder>/<dataset-name>/<format-folder>"

2. Processing text-only data from a single file, in which each line is a document with the following format: <doc_id>\t<doc_content>\n

   java -cp 'dist/segan.jar:dist/lib/*' main.ProcessData --dataset <dataset-name> -file --text-data <input-text-folder> --data-folder <data-folder> --format-folder <format-folder>

   Other options:
   --format-file:	By default, all processed data files will be named <dataset-name>.<extension>, e.g., <dataset-name>.dat, <dataset-name>.wvoc etc. If you want to rename the formatted file, use this option.

3. Processing data with continuous response variable. Each document is associated with a single continuous value.

   As (1) and (2) but with an additional argument:
      --response-file <response-file>
   where <response-file> has the following format: <docid>\t<response_value>\n

4. Processing data with discrete label. Each document is associated with a set of discrete labels.

   As (1) and (2) but with an additional argument:
      --label-file <label-file>
   where <label-file> has the following format: <docid>\t<label_1>\t<label_2>\t...<\n>
   
   Other arguments:
   --L: Maximum label vocab size
   --min-label-df: Minimum label frequency

5. Create cross validation data (currently only support data with continuous responses)

   java -cp 'dist/segan.jar:dist/lib/*' main.CreateCrossValidationFolds --dataset <dataset-name> --data-folder <data-folder> --format-folder <format-folder> --output <cross-validation-folder> --num-folds <number-of-folds> --tr2dev-ratio <training-to-development ratio> --num-classes <number-of-discretized-classes>

   *** Note: --num-classes is the number of classes that the response variable is discretized into, which is used for stratified sampling. For example, if --num-classes = 3, the response variables are discretized into 3 classes, and CreateCrossValidationFolds will try to preserve the distributions of these 3 classes in training, development and test sets. Default, --num-classes = 1.

=== Run SLDA ===
1. Run SLDA Gibbs sampler with the default setting
   java -cp 'dist/segan.jar:dist/lib/*:/fs/clip-ml/gurobi502/linux64/lib/gurobi.jar' sampler.supervised.regression.SLDA --dataset <dataset-name> --data-folder <data-folder> --format-folder <format-folder> --output <result-folder> --K <number-of-topics> -z

2. Run SLDA Gibbs sampler for cross validation
   a) To train on training data
   
   java -cp 'dist/segan.jar:dist/lib/*:/fs/clip-ml/gurobi502/linux64/lib/gurobi.jar' sampler.supervised.regression.SLDA --dataset <dataset-name> --data-folder <data-folder> --format-folder <format-folder> --output <result-folder> --K <number-of-topics> --cv-folder <cross-validation-folder> --num-folds <number-of-folds> --run-mode train -z

   b) To test, use "--run-mode test" instead of "--run-mode train"

   c) To do both training and test, use "--run-mode train-test" instead

   *** Notes:
   + --fold: to specify a single fold that you want to run on
   + -z: to perform z-normalization on the response variable
   + -v: verbose
   + -d: debug

=== Run SHLDA ===
1. Run SHLDA
   java -cp 'dist/segan.jar:dist/lib/*:/fs/clip-ml/gurobi502/linux64/lib/gurobi.jar' sampler.supervised.regression.SHLDA --dataset <dataset-name> --data-folder <data-folder> --format-folder <format-folder> --output <result-folder> -z

2. Run SHLDA for cross validation
   java -cp 'dist/segan.jar:dist/lib/*:/fs/clip-ml/gurobi502/linux64/lib/gurobi.jar' sampler.supervised.regression.SHLDA --dataset <dataset-name> --data-folder <data-folder> --format-folder <format-folder> --output <result-folder> --cv-folder <cross-validation-folder> --num-folds <number-of-folds> --run-mode <run-mode> -z

=== Examples to process and run models with the accompanied amazon data ===
* Process data
  java -Xmx4096M -Xms4096M -cp 'dist/segan.jar:dist/lib/*' main.ProcessData --dataset amazon-data -file --text-data demo/amazon-data/raw/text.txt --response-file demo/amazon-data/raw/response.txt --data-folder demo --format-folder format-response --u 5 --b 3 --bs 5 -s -l --V 5000

* Create cross validation
  java -Xmx4096M -Xms4096M -cp 'dist/segan.jar:dist/lib/*' main.CreateCrossValidationFolds --dataset amazon-data --data-folder demo --format-folder format-response --num-folds 5 --tr2dev-ratio 0.8 --num-classes 3 --output demo/amazon-data/format-response/crossvalidation-5-0.8/

* Run SLDA on the cross validation data
  java -cp 'dist/segan.jar:dist/lib/*:/fs/clip-ml/gurobi502/linux64/lib/gurobi.jar' sampler.supervised.regression.SLDA --dataset amazon-data --data-folder demo --format-folder format-response --output demo/amazon-data/format-response/model --K 25 --cv-folder demo/amazon-data/format-response/crossvalidation-5-0.8 --num-folds 5 --run-mode train-test -v -z

* Run SHLDA on the cross validation data
  java -cp 'dist/segan.jar:dist/lib/*:/fs/clip-ml/gurobi502/linux64/lib/gurobi.jar' sampler.supervised.regression.SHLDA --dataset amazon-data --data-folder demo --format-folder format-response --output demo/amazon-data/format-response/model --cv-folder demo/amazon-data/format-response/crossvalidation-5-0.8 --num-folds 5 --run-mode train-test -v -z
