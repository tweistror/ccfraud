class Preprocess_mnist:
    def __init__(self):
        self.columns = None

        self.scaler = None
        self.pca = None
        self.train_test_dimensions = None
        self.scale_number = 255

    def set_train_test_dimensions(self, train_test_dimensions):
        self.train_test_dimensions = train_test_dimensions

    def preprocess(self, x_sv_train, x_usv_train, x_test):
        x_sv_train = x_sv_train.astype('float32')
        x_usv_train = x_usv_train.astype('float32')
        x_test = x_test.astype('float32')

        x_sv_train /= self.scale_number
        x_usv_train /= self.scale_number
        x_test /= self.scale_number

        return x_sv_train, x_usv_train, x_test

    def inverse_preprocessing(self, data):
        data = data.astype('float32')

        data *= self.scale_number

        return data
