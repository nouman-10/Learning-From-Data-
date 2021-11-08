
### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The libraries required for the successful execution of this code are mentioned in requirements.txt. In order to install all the libraries:
`pip install -r requirements.txt`

## Project Motivation<a name="motivation"></a>


In this project, we tackle the problem of text classification, specifically subject classification using articles from several newspapers. The data contains articles about climate change that have been published in the period around the first 25 Conference of the Parties [1](https://unfccc.int/process/bodies/supreme-bodies/conference-of-the-parties-cop) (COP1 until 24, there was a 6a meeting as well). The articles have been collected from Lexis Nexis [2](https://www.nexisuni.com/) based on a set of keywords and newspapers. 

We tried various machine learning algorithms such as Naive Bayes, Support Vector Machines, Random Forest etc along with Long-Short Term Memory (LSTMs) and BERT models. 
## File Descriptions <a name="files"></a>

- `preprocess.py` takes as input a bunch of json files (or a single json file for testing) and preprocesses the data to extract the required features and labels from it. To run it for a folder containing json files, run `python preprocess.py -i=./data/COP.filt3.sub/` where `./data/COP.filt3.sub/` contains the files which is the default as well. Run `python preprocess.py -t=./data/./data/COP25.filt3.sub.json` for a single test json file.

- `model.py` takes as input the files stored after preprocessing and runs the best baseline model (which you can download from [here](https://drive.google.com/file/d/1bEj0FQ3DZBrJfZkmmsgMhRYXXRmmNS0Q/view?usp=sharing) and should be saved in the `data/` folder) and saves the model/results. It also contains other code for LSTMs and BERT models. To run, simple run the command `python model.py`. It has an optional parameter for a different data directory as well but the files should be named `train.json`, `dev.json`, and `test.json`.

- `predict.py` predicts on a test file using the saved model. You can optionally provide the path for the test file as well as the model.

- `pipeline.sh` automates all the above by first installing all the required libraries. Then taking a test file and preprocessing it, then using the saved baseline model to predict on it. You can run it as `pipeline.sh ./data/COP24.filt3.sub.json`



