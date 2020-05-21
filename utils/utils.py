def format_dataset_string(dataset_string):
    if dataset_string == "paysim":
        return 'PaySim'
    if dataset_string == "paysim-custom":
        return 'PaySim Custom'
    elif dataset_string == "ccfraud":
        return 'PaySim'
    elif dataset_string == "ieee":
        return 'IEEE Kaggle'
    elif dataset_string == "nslkdd":
        return 'NSL-KDD'
    elif dataset_string == "saperp-ek":
        return 'Synthetic SAP-ERP EK'
    elif dataset_string == "saperp-vk":
        return 'Synthetic SAP-ERP VK'
    elif dataset_string == "mnist":
        return 'MNIST'
    elif dataset_string == "cifar10":
        return 'CIFAR10'
