import yaml


class Parameters(object):
    def __init__(self, dataset_string):
        with open(f'./data/{dataset_string}/config.yaml', 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print('Error while trying to load YAML-file')

    def get_main_parameters(self, cross_validation_count):
        factor = 0 if cross_validation_count < 2 else cross_validation_count - 1
        parameters = self.config['parameters']

        usv_train_size = parameters['usv_train_size']
        sv_train_size = parameters['sv_train_size']
        sv_train_frauds = sv_train_size * parameters['sv_train_fraud_percentage']
        test_benign = parameters['test_benign_count'] - factor * sv_train_frauds
        test_fraud = parameters['test_fraud_count'] - factor * sv_train_frauds

        return int(usv_train_size), int(sv_train_size), int(sv_train_frauds), int(test_benign), int(test_fraud)

    def get_path(self):
        return self.config['path']

    def get_ocan_parameters(self):
        ocan_parameters = {
            'gan': self.config['parameters']['ocan'],
            'ae': self.config['parameters']['ocan_ae'],
        }
        return ocan_parameters

    def get_autoencoder_parameters(self):
        return self.config['parameters']['autoencoder']

    def get_vae_parameters(self):
        return self.config['parameters']['vae']

    def get_rbm_parameters(self):
        return self.config['parameters']['rbm']

    def get_denoising_autoencoder_parameters(self):
        return self.config['parameters']['dae']

    def get_saperp_mode(self):
        return self.config['mode']

    def get_mnist_mode(self):
        return self.config['mode']

    def get_cifar10_mode(self):
        return self.config['mode']

    def get_all_parameters(self):
        return self.config['parameters']
