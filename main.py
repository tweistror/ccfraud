import sys
import argparse
from argparse import RawTextHelpFormatter

parser = argparse.ArgumentParser(description='Tool for testing various machine learning methods on different datasets',
                                 formatter_class=RawTextHelpFormatter)
parser.add_argument("--dataset", required=True, choices=["paysim", "ccfraud"], help="Dataset")
parser.add_argument("--method", choices=["sk-svm", "sk-knn", "sk-mlp", "sk-nb", "sk-rf", "sk-lr", "xgb"],
                    required=True, help="Machine learning method used for classification")
parser.add_argument("--mode", required=True, choices=["baseline", "solo"], help='''Execution mode: 
`baseline` for comparison to other baseline methods
`solo` for executing the chosen method only''')


