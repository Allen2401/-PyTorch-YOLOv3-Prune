# import tensorflow as tf
#
#
# class Logger(object):
#     def __init__(self, log_dir):
#         """Create a summary writer logging to log_dir."""
#         self.writer = tf.summary.FileWriter(log_dir)
#
#     def scalar_summary(self, tag, value, step):
#         """Log a scalar variable."""
#         summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
#         self.writer.add_summary(summary, step)
#
#     def list_of_scalars_summary(self, tag_value_pairs, step):
#         """Log scalar variables."""
#         summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value) for tag, value in tag_value_pairs])
#         self.writer.add_summary(summary, step)
from tensorboardX import SummaryWriter
import os
from datetime import datetime
import time

class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        timestamp = datetime.fromtimestamp(time.time()).strftime('%m%d-%H:%M')
        self.writer = SummaryWriter(log_dir)

    def list_of_scalars_summary(self, prefix, strm,tag_value_pairs, step):
        """Log scalar variables."""
        for tag, value in tag_value_pairs:
            self.writer.add_scalars(prefix+'/'+tag, {strm:value}, step)