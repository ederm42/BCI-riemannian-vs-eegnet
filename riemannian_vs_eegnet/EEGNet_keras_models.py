from arl_eegmodels_master.EEGModels import EEGNet, DeepConvNet, EEGNet_SSVEP, ShallowConvNet

# see:
# https://queirozf.com/entries/scikit-learn-pipeline-examples#keras-model


def create_eegnet_default_model(chans, samples, nb_classes):
    # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other
    # model configurations may do better, but this is a good starting point)
    model = EEGNet(nb_classes=nb_classes, Chans=chans, Samples=samples,
                   dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16,
                   dropoutType='Dropout')

    # compile the model and set the optimizers
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


def create_eegnet_deepconvnet_model(chans, samples, nb_classes):
    model = DeepConvNet(nb_classes=nb_classes, Chans=chans, Samples=samples,
                        dropoutRate=0.5)

    # compile the model and set the optimizers
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


def create_eegnet_shallow_model(chans, samples, nb_classes):
    model = ShallowConvNet(nb_classes=nb_classes, Chans=chans, Samples=samples,
                           dropoutRate=0.5)

    # compile the model and set the optimizers
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


def create_eegnet_ssvep_model(chans, samples, nb_classes):
    model = EEGNet_SSVEP(nb_classes=nb_classes, Chans=chans, Samples=samples,
                         dropoutRate=0.5, kernLength=256, F1=96,
                         D=1, F2=96, dropoutType='Dropout')

    # compile the model and set the optimizers
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model
