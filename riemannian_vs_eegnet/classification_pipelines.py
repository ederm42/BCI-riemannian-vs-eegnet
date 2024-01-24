from mne.decoding import CSP
from pyriemann.classification import MDM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from pyriemann.estimation import Covariances, XdawnCovariances, ERPCovariances, HankelCovariances, CospCovariances, Shrinkage
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from keras.wrappers.scikit_learn import KerasClassifier

from moabb.pipelines import ExtendedSSVEPSignal

from EEGNet_keras_models import create_eegnet_default_model, \
    create_eegnet_deepconvnet_model, \
    create_eegnet_shallow_model, \
    create_eegnet_ssvep_model


# #
# MI
# #
def get_mi_pipelines(**kwargs):
    pipelines = {}

    # from http://moabb.neurotechx.com/docs/auto_tutorials/tutorial_3_benchmarking_multiple_pipelines.html
    pipelines["CSP+LDA"] = make_pipeline(
        CSP(n_components=8),
        LDA()
    )

    # from http://moabb.neurotechx.com/docs/auto_tutorials/tutorial_3_benchmarking_multiple_pipelines.html
    pipelines["TGSP+SVM"] = make_pipeline(
        Covariances("oas"),
        TangentSpace(metric="riemann"),
        SVC(kernel="linear")
    )

    # from F Lotte: A review of classification algorithms for EEG-based brain-computer interfaces: a 10 year update
    pipelines["TGSP+LDA"] = make_pipeline(
        Covariances("oas"),
        TangentSpace(metric="riemann"),
        LDA()
    )

    # from F Lotte: A review of classification algorithms for EEG-based brain-computer interfaces: a 10 year update
    # from http://moabb.neurotechx.com/docs/auto_tutorials/tutorial_3_benchmarking_multiple_pipelines.html
    pipelines["RMDM"] = make_pipeline(
        Covariances("oas"),
        MDM(metric="riemann")
    )
    return pipelines


# #
# ERP
# #
def get_erp_pipelines(**kwargs):
    # things to look:
    # Bertrand Rivet, Antoine Souloumiac, Virginie Attina, and Guillaume Gibert.
    # xDAWN algorithm to enhance evoked potentials: application to brain–computer interface.
    # IEEE Transactions on Biomedical Engineering, 56(8):2035–2043, 2009. doi:10.1109/TBME.2009.2012869.
    #
    # https://hal.archives-ouvertes.fr/hal-00602700/file/2010-09_LVA.pdf

    # set up sklearn pipeline
    pipelines = {}

    # https://github.com/vlawhern/arl-eegmodels/blob/master/examples/ERP.py
    n_components = 2  # pick some components
    pipelines["TGSP+LR"] = make_pipeline(
        XdawnCovariances(n_components),
        TangentSpace(metric='riemann'),
        LogisticRegression()
    )

    # from F Lotte: A review of classification algorithms for EEG-based brain-computer interfaces: a 10 year update
    pipelines["ERPCov+RMDM"] = make_pipeline(
        ERPCovariances(),
        MDM(metric="riemann")
    )

    # from F Lotte: A review of classification algorithms for EEG-based brain-computer interfaces: a 10 year update
    pipelines["XCov+RMDM"] = make_pipeline(
        XdawnCovariances(n_components),
        MDM(metric="riemann")
    )
    return pipelines


# #
# SSVEP
# #
def get_ssvep_pipelines(**kwargs):
    # Possible literature:
    # Riemannian classification for SSVEP based BCI: offline versus online implementations
    # https://hal.uvsq.fr/hal-01710089/document

    # Bertrand Rivet, Antoine Souloumiac, Virginie Attina, and Guillaume Gibert.
    # xDAWN algorithm to enhance evoked potentials: application to brain–computer interface.
    # IEEE Transactions on Biomedical Engineering, 56(8):2035–2043, 2009. doi:10.1109/TBME.2009.2012869.
    n_components = 2  # pick some components

    pipelines = {}
    # F Lotte: A review of classification algorithms for EEG-based brain-computer interfaces: a 10 year update
    # https://hal.archives-ouvertes.fr/hal-00602700/file/2010-09_LVA.pdf
    pipelines["Cov+RMDM"] = make_pipeline(
        # ExtendedSSVEPSignal(),
        Covariances("oas"),
        MDM(metric="riemann")
    )
    pipelines["Cov+LogMDM"] = make_pipeline(
        # ExtendedSSVEPSignal(),
        Covariances("oas"),
        MDM(metric="logeuclid")
    )
    pipelines["TGSP+LR"] = make_pipeline(
        # ExtendedSSVEPSignal(),
        Covariances("oas"),
        TangentSpace(metric="riemann"),
        LogisticRegression(solver="lbfgs", multi_class="auto"),
    )


    # http://moabb.neurotechx.com/docs/auto_examples/plot_cross_subject_ssvep.html
    # pipelines["RG+LogReg"] = make_pipeline(
    #    ExtendedSSVEPSignal(),
    #    Covariances(estimator="lwf"),
    #    TangentSpace(),
    #    LogisticRegression(solver="lbfgs", multi_class="auto"),
    #)

    return pipelines


# #
# SSVEP
# #
def get_ssvep_fb_pipelines(**kwargs):
    # Possible literature:
    # Riemannian classification for SSVEP based BCI: offline versus online implementations
    # https://hal.uvsq.fr/hal-01710089/document

    # Bertrand Rivet, Antoine Souloumiac, Virginie Attina, and Guillaume Gibert.
    # xDAWN algorithm to enhance evoked potentials: application to brain–computer interface.
    # IEEE Transactions on Biomedical Engineering, 56(8):2035–2043, 2009. doi:10.1109/TBME.2009.2012869.
    n_components = 2  # pick some components

    pipelines = {}
    # F Lotte: A review of classification algorithms for EEG-based brain-computer interfaces: a 10 year update
    # https://hal.archives-ouvertes.fr/hal-00602700/file/2010-09_LVA.pdf
    pipelines["Cov+RMDM"] = make_pipeline(
        ExtendedSSVEPSignal(),
        Covariances("oas"),
        MDM(metric="riemann")
    )
    pipelines["Cov+LogMDM"] = make_pipeline(
        ExtendedSSVEPSignal(),
        Covariances("oas"),
        MDM(metric="logeuclid")
    )
    pipelines["TGSP+SVM"] = make_pipeline(
        ExtendedSSVEPSignal(),
        Covariances("oas"),
        TangentSpace(metric="riemann"),
        SVC(kernel="linear")
    )
    # http://moabb.neurotechx.com/docs/auto_examples/plot_cross_subject_ssvep.html
    pipelines["TGSP+LogReg"] = make_pipeline(
        ExtendedSSVEPSignal(),
        Covariances(estimator="oas"),
        TangentSpace(metric="riemann"),
        LogisticRegression(solver="lbfgs", multi_class="auto"),
    )

    # http://moabb.neurotechx.com/docs/auto_examples/plot_cross_subject_ssvep.html
    # pipelines["RG+LogReg"] = make_pipeline(
    #    ExtendedSSVEPSignal(),
    #    Covariances(estimator="lwf"),
    #    TangentSpace(),
    #    LogisticRegression(solver="lbfgs", multi_class="auto"),
    #)

    return pipelines


# #
# EEGNET
# #
def get_eegnet_pipelines(chans, samples, nb_classes, eegnet_types=None):
    if eegnet_types is None:
        eegnet_types = ["default", "deep", "shallow"]

    pipelines = {}
    build_fns = {
        "default": create_eegnet_default_model,
        "default_fb": create_eegnet_default_model,
        "deep": create_eegnet_deepconvnet_model,
        "deep_fb": create_eegnet_deepconvnet_model,
        "shallow": create_eegnet_shallow_model,
        "shallow_fb": create_eegnet_shallow_model,
        "ssvep": create_eegnet_ssvep_model,
        "ssvep_fb": create_eegnet_ssvep_model
    }
    for eegnet_type in eegnet_types:
        # build basic keras model
        pipeline_name = f"EEGNet_{eegnet_type}"
        build_fn = build_fns[eegnet_type]
        clf = KerasClassifier(build_fn=build_fn,
                              chans=chans, samples=samples, nb_classes=nb_classes,  # build_fn arguments
                              verbose=2)
        # clf._estimator_type = 'classifier'

        # special handling for certain pipelines with extra steps
        if "fb" in eegnet_type:
            pipelines[pipeline_name] = make_pipeline(ExtendedSSVEPSignal(), clf)
        else:  # default
            pipelines[pipeline_name] = make_pipeline(clf)

    return pipelines
