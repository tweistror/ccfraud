import argparse


class Parser(object):
    def __init__(self, datasets, methods, baselines):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", required=True, choices=datasets, help="Desired dataset for evaluation")
        parser.add_argument("--method", choices=methods,
                            help="Method for evaluation (no specification will result in no evaluation of any advanced "
                                 "method, all with execute all advanced methods)")
        parser.add_argument("--baselines", choices=baselines, default="both",
                            help="Baselines for evaluation (default is both)")
        parser.add_argument("--v", choices=['0', '1', '2'], default=0,
                            help="Verbosity level (0 = just end results, 1 = "
                                 "some timing information, "
                                 "2 = more timing information)")
        parser.add_argument("--iterations", default="10",
                            help="Desired count the specified methods are executed and evaluated")
        parser.add_argument("--cv", help="Activate crossvalidation with the desired count of train/test-splits")
        parser.add_argument("--oversampling", choices=['y', 'n'], default='n', help="Use oversampling (SMOTE) or not")
        self.parser = parser

    def get_args(self):
        args = self.parser.parse_args()

        return args.dataset, int(args.v), args.method, args.baselines, int(args.iterations), \
            True if args.oversampling == 'y' else False, 1 if args.cv is None else int(args.cv)
