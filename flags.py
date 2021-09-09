from absl import flags

FLAGS = flags.FLAGS


def setup_flags():
    """
    Definition of command line arguments.
    """
    flags.DEFINE_enum(name="dataset", default="cifar10", enum_values=["cifar10", "mnist", "fashion_mnist",
                                                                                 "svhn_cropped", "plant32", "plant64",
                                                                                 "plant96"], help="dataset name")
    flags.DEFINE_enum(name="network", default="wrn-28-2", enum_values=["wrn-28-2", "simple", "vgg16",
                                                                                  "ssdl"],
                      help="network architecture.")
    flags.DEFINE_boolean('weights', help="random or ImageNet pretrained weights", default=False)
    flags.DEFINE_integer("batch_size", help="size of mini-batch", default=100)
    flags.DEFINE_integer('epochs', help="initial training epochs", default=200)
    flags.DEFINE_boolean('semi', help="True: N-labelled training, False: All-labelled training", default=True)
    flags.DEFINE_enum(name="lt", default="cross-entropy", enum_values=["cross-entropy", "triplet", "arcface",
                                                                       "contrastive"],
                      help="loss_type: cross-entropy, triplet,  arcface or contrastive.")
    flags.DEFINE_enum(name="opt", default="adam", enum_values=["adam", "sgd", "rmsprop"],
                      help="optimizer.")
    flags.DEFINE_float('lr', help="learning_rate", default=1e-4)
    # metric learning losses related
    flags.DEFINE_enum('lbl', help="shallow classifiers for labelling for metric learning losses", default="knn",
                      enum_values=["knn", "lda", "rf", "lr"])
    flags.DEFINE_float('margin', help="margin for triplet loss calculation", default=1.0)
    #  self-training related
    flags.DEFINE_boolean("self_training",  help="apply self-training", default=False)
    flags.DEFINE_enum(name="confidence_measure", default="1-nn", enum_values=["1-nn", "llgc"],
                      help="confidence measure for pseudo-label selection.")
    flags.DEFINE_integer('meta_iterations', help="number of meta_iterations", default=25)
    flags.DEFINE_integer('epochs_per_m_iteration', help="number of epochs per meta-iterations", default=200)
    # extras
    flags.DEFINE_string('gpu', help="gpu id", default="0")
    flags.DEFINE_integer('verbose', help="verbose", default=1)
    flags.DEFINE_string('pre', help="prefix for log directory", default='')

