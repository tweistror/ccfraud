# Financial Fraud Detection

## Preparation
* Clone/Download this repository
* Download the desired dataset folder from https://drive.google.com/drive/folders/1ioHeGFWo-4lQZcTf6p4tknG56Z2TQGWE and move the folder into the `data` Folder.
* The `config.yaml`-file in all dataset-folders are used for various configurations and contain some metadata as well

## How to start the program?

The cmd-based tool is started with a simple cmd-command with specific start parameters.

A basic example:
`python main.py --dataset=ccfraud --v=1 --iterations=10 --baselines=both --method=all`

This table presents all available start-parameters

Parameter name  | Required | Default | Choices | Description
------------- | ------------- | ------------ | ------------ | ------------
--dataset  | yes | -- | paysim, paysim-custom, ccfraud, ieee | Desired dataset for evaluation
--method  | no | -- | all, ocan, ocan-ae, ae, rbm, vae | Method for evaluation (no specification will result in no evaluation of any advanced method, `all` with execute all advanced methods)
--baselines | no | both | both, sv, usv, none | Baselines for evaluation (default is both)
--v | no | 0 | 0, 1, 2 | Verbosity level (0 = just print end results, 1 = some timing information, 2 = more timing information & training information)
--iterations    | no | 10 | number (not too high) | Desired count the specified methods are executed and evaluated
--cv    | no | -- | number (not too high) | Activate crossvalidation with the desired count of train/test-splits
--oversampling  | no | n | y, n | Flag for activation of oversampling (default is no)
--seed | no | random | 'random' or a concrete number| Specify a number as concrete seed (default is random)
--plots | no | none | none, roc, pr, both | Curves to be plotted

Further, `--help` is available to show console-based help.

