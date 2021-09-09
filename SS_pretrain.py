from absl import app
from absl import logging
import os
import sys
import signal
import datetime
from absl import flags
from utils.utils import program_duration
import train_utils
import SS_utils

FLAGS = flags.FLAGS
flags.DEFINE_integer("pretraining_epochs", help="pretraining epochs : weight of self-supervised loss", default=200)


def main(argv):
    dt1 = datetime.datetime.now()
    del argv  # not used

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    dso, data_config = train_utils.set_dataset(FLAGS.dataset, FLAGS.lt, FLAGS.semi)
    _, supervised_model, self_supervised_model = SS_utils.get_model(FLAGS.network, data_config,
                                                                                 FLAGS.weights, FLAGS.lt, FLAGS.opt,
                                                                                 FLAGS.lr, ret_self_model=True)
    # set up logging details
    log_dir, log_name = train_utils.get_log_name(FLAGS, data_config, prefix="ss-pretrain-")
    os.makedirs(log_dir, exist_ok=True)
    logging.get_absl_handler().use_absl_log_file(log_name, log_dir)
    logging.get_absl_handler().setFormatter(None)
    print("training logs are saved at: ", log_dir)
    logging.info(FLAGS.flag_values_dict())
    ac = train_utils.log_accuracy(supervised_model, dso, FLAGS.lt, FLAGS.semi, labelling=FLAGS.lbl)
    logging.info("init Test accuracy : {:.2f} %".format(ac))

    def ctrl_c_accuracy():
        ac_ = train_utils.log_accuracy(supervised_model, dso, FLAGS.lt, FLAGS.semi, labelling=FLAGS.lbl)
        logging.info("ctrl_c_accuracy Test accuracy : {:.2f} %".format(ac_))
        print(program_duration(dt1, 'Killed after Time'))

    def exit_gracefully(signum, frame):
        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, original_sigint)
        try:
            if input("\nReally quit? (y/n)> ").lower().startswith('y'):
                ctrl_c_accuracy()
                sys.exit(1)
        except KeyboardInterrupt:
            print("Ok ok, quitting")
            sys.exit(1)

    signal.signal(signal.SIGINT, exit_gracefully)

    print("Pretraining...")
    SS_utils.start_self_supervised_pretraining(self_supervised_model, dso, FLAGS.semi, FLAGS.pretraining_epochs,
                                               FLAGS.batch_size)

    print("Fine-tuning...")
    train_utils.start_training(supervised_model, dso, lt=FLAGS.lt, epochs=FLAGS.epochs, semi=FLAGS.semi,
                               bs=FLAGS.batch_size)

    ac = train_utils.log_accuracy(supervised_model, dso, FLAGS.lt, FLAGS.semi, labelling=FLAGS.lbl)
    logging.info("after training Test accuracy : {:.2f} %".format(ac))

    if FLAGS.self_training:
        # reduce lr by factor of 0.1
        from tensorflow.keras import backend as keras_backend
        keras_backend.set_value(supervised_model.optimizer.learning_rate, FLAGS.lr/10.)

        train_utils.start_self_learning(supervised_model, dso, data_config, FLAGS.lt, FLAGS.confidence_measure,
                                        FLAGS.meta_iterations, FLAGS.epochs_per_m_iteration, FLAGS.batch_size, logging)

    print(program_duration(dt1, 'Total Time taken'))


if __name__ == '__main__':
    from flags import setup_flags
    setup_flags()
    FLAGS.alsologtostderr = True  # also show logging info to std output
    app.run(main)


