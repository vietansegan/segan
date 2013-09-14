1. Compile
   cd <SEGAN_PATH>
   ant jar
   cp -r lib dist/

2. Process data
To process data from multiple files from a folder
   cd <SEGAN_PATH>
   java -cp 'dist/segan.jar:dist/lib/*' main.ProcessData --dataset <dataset-name> --text-data <input-text-folder> --response-file <response-file> --otput-folder <output-folder>

where
- <dataset-name>	The name of the dataset
- <input-text-folder>	The folder that contains input text data, in which each file is one document
- <response-file>	The file contains the response variable, one line for each document
- <output-folder>	The folder where the processed data will be stored

To process data from a single file, each line is a document with the following format: <doc_id>\t<doc_content>\n
   cd <SEGAN_PATH>
   java -cp 'dist/segan.jar:dist/lib/*' main.ProcessData --dataset <dataset-name> -file --text-data <input-text-file> --response-file <response-file> --otput-folder <output-folder>

where
- <dataset-name>        The name of the dataset
- <input-text-file>	The file that contains input text data, in which each line is one document
- <response-file>       The file contains the response variable, one line for each document
- <output-folder>	The folder where the processed data will be stored

For unsupervised models like LDA or HDP, we do not need <response-file>.

For more options for processing data
    java -cp dist/segan.jar main.ProcessData -help

Examples
- To process the toydata (from multiple files in a folder)
    java -cp 'dist/segan.jar:dist/lib/*' main.ProcessData --dataset toydata --text-data toydata/text/ --response-file toydata/response.txt --output-folder toydata/processed-data/

*** Note
- If you see the following, they are expected
Trying to add database driver (JDBC): RmiJdbc.RJDriver - Error, not in CLASSPATH?
Trying to add database driver (JDBC): jdbc.idbDriver - Error, not in CLASSPATH?
Trying to add database driver (JDBC): org.gjt.mm.mysql.Driver - Error, not in CLASSPATH?
Trying to add database driver (JDBC): com.mckoi.JDBCDriver - Error, not in CLASSPATH?
Trying to add database driver (JDBC): org.hsqldb.jdbcDriver - Error, not in CLASSPATH?

- If you have java.lang.OutOfMemoryError: Java heap space, try to increase the memory allocated for Java by using -Xmx and -Xms options (e.g., -Xmx4096M -Xms4096M to use 4G of RAM)

3. Run LDA
To run LDA Gibbs sampler with the default hyperparameter setting
   java -cp 'dist/segan.jar:dist/lib/*' main.runLDA --dataset <dataset-name> --folder <processed-data-folder> --K <number-of-topics> --output <output-folder>

where
- <dataset-name>		The name of the dataset
- <processed-data-folder>	The folder storing the processed data (i.e., the output folder of main.ProcessData above)
- <number-of-topics>		    Number of topics
- <output-folder> 		    	   The output folder

For more options for running LDA
    java -cp 'dist/segan.jar:dist/lib/*' main.RunLDA -help

Example to run LDA on the processed toydata
    java -cp 'dist/segan.jar:dist/lib/*' main.RunLDA --dataset toydata --folder toydata/processed-data/ --K 10 --output toydata/lda/ -v -report 20

4. Run SLDA
To run SLDA Gibbs sampler with the default hyperparameter setting
   java -cp 'dist/segan.jar:dist/lib/*' main.runSLDA --dataset <dataset-name> --folder <processed-data-folder> --K <number-of-topics> --output <output-folder>

For more options for running SLDA
   java -cp 'dist/segan.jar:dist/lib/*' main.RunSLDA --help

Examples
- To run SLDA on the processed toydata
   java -cp 'dist/segan.jar:dist/lib/*' main.RunSLDA --dataset toydata --folder toydata/processed-data/ --K 10 --output toydata/slda/ -v -report 20

5. Run SHLDA
To run SHLDA Gibbs sampler with the default settings
   java -cp 'dist/segan.jar:dist/lib/*' main.runSHLDA --dataset <dataset-name> --folder <processed-data-folder> --output <output-folder>

For more options for running SHLDA
    java -cp 'dist/segan.jar:dist/lib/*' main.RunSHLDA --help

Examples
- To run SHLDA on the processed toydata
   java -cp 'dist/segan.jar:dist/lib/*' main.RunSHLDA --dataset toydata --folder toydata/processed-data/ --output toydata/shlda/ -v

6. Create cross validation data
   java -cp 'dist/segan.jar:dist/lib/*' main.CreateCrossValidationFolds --dataset <dataset-name> --folder <processed-data-folder> --output <cross-validation-folder> --num-folds <number-of-folds> --tr2dev-ratio <training-to-development ratio> --num-classes <number-of-discretized-classes>

*** Note: --num-classes is the number of classes that the response variable is discretized into, which is used for stratified sampling. For example, if --num-classes = 3, the response variables are discretized into 3 classes, and CreateCrossValidationFolds will try to preserve the distributions of these 3 classes in training, development and test sets. Default, --num-classes = 1.

Example:
   java -Xmx4096M -Xms4096M -cp 'dist/segan.jar:dist/lib/*' main.CreateCrossValidationFolds --dataset amazon-data --folder amazon-data/processed-data/ --num-folds 5 --tr2dev-ratio 0.8 --num-classes 3 --output amazon-data/crossvalidation-5-0.8/

7. Run SLDA with cross validation data
7a. To train on training data of all folds
   java -cp 'dist/segan.jar:dist/lib/*' main.RunSLDA --dataset <dataset-name> --folder <processed-data-folder> --K <number-of-topics> --output <output-folder> --cv-folder <cross-validation-folder> --num-folds <number-of-folds> --run-mode train

Example:
   java -Xmx4096M -Xms4096M -cp 'dist/segan.jar:dist/lib/*' main.RunSLDA --dataset amazon-data --folder amazon-data/processed-data/ --K 25 --output amazon-data/slda/ -v -report 1 --burnIn 250 --maxIter 500 --sampleLag 25 -s -v -d --cv-folder amazon-data/crossvalidation-5-0.8/ --num-folds 5 --run-mode train

7b. To test on test data of all folds after training: use the same command line except for using 'test' --run-mode

Example:
   java -Xmx4096M -Xms4096M -cp 'dist/segan.jar:dist/lib/*' main.RunSLDA --dataset amazon-data --folder amazon-data/processed-data/ --K 25 --output amazon-data/slda/ -v -report 1 --burnIn 50 --maxIter 100 --sampleLag 10 -s -v -d --cv-folder amazon-data/crossvalidation-5-0.8/ --num-folds 5 --run-mode test

*** Note:
A) During test time, multiple models outputted during training time (in fold ~/report) will be used, each will run for --maxIter iterations. Thus, if --maxIter is big, this will take a while :).

B) Running test will output various result files which corresponds to different ways of using the Gibbs samples for prediction. The evaluation measurements are Pearson's Correlation Coefficient, Mean Squared Error and R-squared. The result files are:
- single-final.txt: Using the final predicted values of each individual model to predict
- single-avg.txt: Using the average predicted values of each individual model to predict
- multiple-final.txt: Using the final predicted values of each individual model, and average over them to predict
- multiple-avg.txt: Using the average predicted values of each individual model, and average over them to predict
More details to come, but if you don't really care about various ways to do prediction, just use the result in 'multiple-avg.txt', which usually gives the best performance.

C) To run on a particular fold, use the "--fold" option. For example, "--fold 0" to run only on fold 0. If no "--fold" option is available, all available folds will be used.

7c. To run both train and test, use 'train-test' --run-mode

Example
    java -Xmx4096M -Xms4096M -cp 'dist/segan.jar:dist/lib/*' main.RunSLDA --dataset amazon-data --folder amazon-data/processed-data/ --K 25 --output amazon-data/slda/ -v -report 1 --burnIn 10 --maxIter 20 --sampleLag 5 -s -v -d --cv-folder amazon-data/crossvalidation-5-0.8/ --num-folds 5 --run-mode train-test


*** Example of a pipeline to run SLDA on the amazon reviews
# compile
cd <SEGAN_PATH>
ant jar
cp -r lib dist/

# process data
java -Xmx4096M -Xms4096M -cp 'dist/segan.jar:dist/lib/*' main.ProcessData --dataset amazon-data -file --text-data amazon-data/text.txt --response-file amazon-data/response.txt --output-folder amazon-data/processed-data/ --u 5 --b 3 --bs 5 -s -l --V 5000

# run SLDA (with z-normalization on the response variable)
java -Xmx4096M -Xms4096M -cp 'dist/segan.jar:dist/lib/*' main.RunSLDA --dataset amazon-data --folder amazon-data/processed-data/ --K 25 --output amazon-data/slda/ -v -report 1 --burnIn 250 --maxIter 500 --sampleLag 25 -s

# create cross validation data
java -Xmx4096M -Xms4096M -cp 'dist/segan.jar:dist/lib/*' main.CreateCrossValidationFolds --dataset amazon-data --folder amazon-data/processed-data/ --num-folds 5 --tr2dev-ratio 0.8 --num-classes 3 --output amazon-data/crossvalidation-5-0.8/

# train and test using SLDA on only fold 0
java -Xmx4096M -Xms4096M -cp 'dist/segan.jar:dist/lib/*' main.RunSLDA --dataset amazon-data --folder amazon-data/processed-data/ --K 25 --output amazon-data/slda/ -v -report 1 --burnIn 250 --maxIter 500 --sampleLag 25 -s -v -d --cv-folder amazon-data/crossvalidation-5-0.8/ --num-folds 5 --run-mode train-test --fold 0
