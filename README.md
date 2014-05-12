# Compile

## Dependencies
Most of `segan`'s dependencies are provided in the `lib` folder. In addition, `segan` also uses `gurobi` for solving optimization problems. To run models that uses `gurobi`, you need to setup `gurobi` on the machine that you will be running. To set up `gurobi`, do the following:

- Go to gurobi.com and create an academic account (it's free).
- Add the following to your `.bashrc` file
```    
    export GUROBI_HOME=<YOUR_GUROBI_PATH>
    export PATH=$PATH:$GUROBI_HOME/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GUROBI_HOME/lib
```
- Log in to `gurobi` on its website, click Download -> Licenses (-> Free Academic) -> Agree and get a license. The website will give you a string like `grbgetkey ####` where `####` is a string of hex number.
- From shell on the machine that you want to install gurobi, run `grbgetkey ####` and choose the location you want to put the license file. If the license file is not in your home directory, you need to put this into your `.bashrc` or `.bash_profile`.
```
    export GRB_LICENSE_FILE=<license_file_directory>
```
- Change the `gurobi.dir` in `build.xml` to `<YOUR_GUROBI_PATH>`

## Build & Compile
1. To compile `ant compile`
2. To build jar file `ant jar`
3. To make a clean build `ant clean-build`

Take a look at the `build.xml` for more options.

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
 - `<format-file>`(optional): By default, all processed data files will be named `<dataset-name>.<extension>`, e.g., `<dataset-name>.dat`, `<dataset-name>.wvoc` etc. If you want to rename these formatted files to `<format-file>.dat`, `<format-file>.wvoc` etc, use this option.

#### Notes

 - For other arguments, use `-help`. For example: `java -cp dist/segan.jar data.TextDataset -help`
 - For each document in the input folder, the filename is used as its ID. If the filename has `txt` as the file extension, `txt` will be discarded. For example, a document stored in file `doc-1.txt` will have `doc-1` as its ID, whereas a document stored in file `doc-2.dat` will have `doc-2.dat` as its ID.

### Processing single file

To process text-only data stored in a single file, where each line corresponds to a document with the following format `<doc_id>\t<doc_content>\n`, use the same command as before but point `--text-data` to where the text data file is.
 - `<input-text-file>`:	The file that contains input text data, in which each line is one document.

## Text data with continuous response variable

This is to process a collection of documents, each of which is associated with a single continuous response variable. This is done similarly as above, but using `data.ResponseTextDataset` with an additional argument `<response-file>` which contains the value of the response variable. The `<response-file>` has the following format: `<docid>\t<response_value>\n`.

```
java -cp 'dist/segan.jar:dist/lib/*' data.ResponseTextDataset --dataset <dataset-name> --text-data <input-text-file> --data-folder <data-folder> --format-folder <format-folder> --run-mode process --response-file <response-file>
```

Working cmd to process the amazon data included in the `demo` folder
```    
java -cp 'dist/segan.jar:dist/lib/*' data.ResponseTextDataset --dataset amazon --text-data demo/amazon-data/raw/text.txt --data-folder demo --format-folder format --run-mode process --response-file demo/amazon-data/raw/response.txt --u 5 --b 10 --bs 5 -s -l --V 1000 -v -d
```
## Create cross validation data

Currently only support data with continuous responses

```
java -cp 'dist/segan.jar:dist/lib/*' data.ResponseTextDataset--dataset <dataset-name> --text-data <input-text-folder> --response-file <response_value> --data-folder <data-folder> --cv-folder <cross-validation-folder> --num-folds <number-of-folds> --tr2dev-ratio <training-to-development-ratio> --num-classes <number-of-discretized-classes> --run-mode cross-validation
```

 - `<cv-folder>`: Directory of the folder to contain the processed cross-validation data.
 - `<num-folds>`: The number of folds
 - `<tr2dev-ratio>`: Training-to-development ratio: the ratio between the number of training instances to the number of development instances.
 - `<num-classes>` is the number of classes that the response variable is discretized into, which is used for stratified sampling. For example, if `<num-classes> = 3`, the response variables are discretized into 3 classes, and CreateCrossValidationFolds will try to preserve the distributions of these 3 classes in training, development and test sets. Default, `<num-classes> = 1`.

## Process training/test data separately

Given a training/test split, this will process the training data and use the processed word vocabulary to process the test data.

To process the training data, use `--run-mode process` as above. For example,

    java -cp 'dist/segan.jar:dist/lib/*' data.TextDataset --dataset <dataset-name> --text-data <input-text-folder> --data-folder <data-folder> --format-folder <format-folder> --format-file <format-file> --run-mode process

To process the test data with the processed word vocabulary, use `--word-vocab-file` to specify the directory of the file containing word vocabulary (i.e., `<output-folder>/<dataset-name>/<format-folder>/<format-file>.wvoc`).

 
## Text data with categorical response variable

Under construction.

# Input Data Format

Under construction.

# Run SHLDA

## Train

    java  -cp 'dist/segan.jar:dist/lib/*:/fs/clip-ml/gurobi502/linux64/lib/gurobi.jar' sampler.supervised.regression.SHLDA --dataset <dataset-name> --data-folder <data-folder> --format-folder <format-folder> --output <result-folder> --run-mode train 

 - `<dataset>`, `<data-folder>`, `<format-folder>`: See above
 - `<format-file>`: Name of input data files without extension.
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


# Run SLDA

## Train

```
java -Xmx10000M -Xms10000M -cp 'dist/segan.jar:lib/*:<GUROBI_JAR_FILE>' sampler.supervised.regression.slda.SLDA --dataset <dataset-name> --data-folder <data-folder> --format-folder <format-folder> --format-file <format-file> --output <result-folder> --burnIn <burn-in> --maxIter <number-of-iterations> --sampleLag <sampler-lag> --K <number-of-topics> -train
```

 - `<dataset>`, `<data-folder>`, `<format-folder>`: See above
 - `<format-file>`: Name of input data files without extension. Default: same as <dataset>. Specify this if when run on different sets of data during cross validation.
 - `<output>`: Directory of the output folder
 - `<burn-in>`: Number of burn-in iterations. Default: 500.
 - `<number-of-iterations>`: Total number of iterations. Default: 1000.
 - `<sample-lag>`: Number of iterations between each model outputed. Default: 50.
 - `<number-of-topics>`: Number of topics. Default: 50

Working example (node: replace <GUROBI_JAR_PATH> with the actual direction to `gurobi.jar` on your machine).

<code>
java -Xmx10000M -Xms10000M -cp 'dist/segan.jar:lib/*:<GUROBI_JAR_PATH>' sampler.supervised.regression.slda.SLDA --dataset amazon-data --data-folder demo --format-folder format --output demo/amazon-data/format-models --burnIn 25 --maxIter 50 --sampleLag 5 --K 25 --alpha 0.1 --beta 0.1 --rho 1 --sigma 10 --init random -train  -z -v -d
</code>

## Test

```
java -Xmx10000M -Xms10000M -cp 'dist/segan.jar:lib/*:<GUROBI_JAR_FILE>' sampler.supervised.regression.slda.SLDA --dataset <dataset-name> --data-folder <data-folder> --format-folder <format-folder> --format-file <format-file> --output <result-folder> --burnIn <burn-in> --maxIter <number-of-iterations> --sampleLag <sampler-lag> --K <number-of-topics> -test --prediction-folder <prediction-folder> --evaluation-folder <evaluation-folder>
```

 - `<prediction-folder>`: Folder to store predicted values made by each learned model
 - `<evaluation-folder>`: Folder to store evaluation results. Evaluation metrics include Pearson correlation coefficient, mean square error (MSE), mean absolute error (MAE), R-squared, predictive R-squared.

## Run cross-validation

Under construction
