from __future__ import division, print_function

import os
import argparse
import configparser
import logging



definitions = [
    # model               type   default help
    ('model',            (str,   'RNN', "Model: fully-connected multi-layer RNN")),
    ('feature_dim',      (int,   4,     "Number of features")),
    ('SEQ_LENGTH',       (int,   30,      "Length of sequence.")),
    ('units',            (int, [50 20 50],    "# of units in each layer.")),
    ('RNNobj',           (str,   'RNN', "Either RNN or LSTM cells.")),
    ('dropout',          (float, 0.2,    "Rate for dropout of activation units.")),


    # training
    ('epochs',           (int,   100,     "Number of epochs to train.")),
    ('batch-size',       (int,   64,     "Mini-batch size for training.")),
    ('train_size',       (float, 0.5,    "Percentage of data to train on.")),
    ('validate_size',    (float, 0.25,    "Percentage of data to validate on.")),
    ('test_size',        (float, 0.25,    "Percentage of data to hold out for testing.")),
    ('shuffle',          {'default': False, 'action': 'store_true',
                          'help': "Shuffle samples before each training epoch."}),
    ('RANDOM_STATE',             (int,   None,   "Seed for numpy RandomState")),

    # files
    ('datadir',          (str,   '.',    "Directory containing data.")),
    ('outdir',           (str,   '.',    "Directory to write output data.")),
]



def update_from_configfile(args, default, config, section, key):
    # Point of this function is to update the args Namespace.
    value = config.get(section, key)
    if value == '' or value is None:
        return

    # Command-line arguments override config file values
    if getattr(args, key) != default:
        return

    # Config files always store values as strings -- get correct type
    if isinstance(default, bool):
        value = config.getboolean(section, key)
    elif isinstance(default, int):
        value = config.getint(section, key)
    elif isinstance(default, float):
        value = config.getfloat(section, key)
    elif isinstance(default, str):
        value = config.get(section, key)
    elif isinstance(default, list):
        # special case (HACK): loss-weights is list of floats
        string = config.get(section, key)
        value = [float(x) for x in string.split()]
    elif default is None:
        # values which aren't initialized
        getter = getattr(config, noninitialized[key])
        value = getter(section, key)
    setattr(args, key, value)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train RNN on Behavioral Data")

    for argname, kwargs in definitions:
        d = kwargs
        if isinstance(kwargs, tuple):
            d = dict(zip(['type', 'default', 'help'], kwargs))
        parser.add_argument('--' + argname, **d)

    # allow user to input configuration file
    parser.add_argument(
        'configfile', nargs='?', type=str, help="Load options from config "
        "file (command line arguments take precedence).")

    args = parser.parse_args()

    if args.configfile:
        logging.info("Loading options from config file: {}".format(args.configfile))
        config = configparser.ConfigParser(
            inline_comment_prefixes=['#', ';'], allow_no_value=True)
        config.read(args.configfile)
        for section in config:
            for key in config[section]:
                if key not in args:
                    raise Exception("Unknown option {} in config file.".format(key))
                update_from_configfile(args, parser.get_default(key),
                                       config, section, key)

    for k,v in vars(args).items():
        logging.info("{:20s} = {}".format(k, v))

return args
