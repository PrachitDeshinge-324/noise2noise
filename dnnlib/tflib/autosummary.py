# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Helper for adding automatically tracked values to Tensorboard.

Autosummary creates an identity op that internally keeps track of the input
values and automatically shows up in TensorBoard. The reported value
represents an average over input components. The average is accumulated
constantly over time and flushed when save_summaries() is called.

Notes:
- The output tensor must be used as an input for something else in the
  graph. Otherwise, the autosummary op will not get executed, and the average
  value will not get accumulated.
- It is perfectly fine to include autosummaries with the same name in
  several places throughout the graph, even if they are executed concurrently.
- It is ok to also pass in a python scalar or numpy array. In this case, it
  is added to the average immediately.
"""

import numpy as np
import tensorflow as tf
import os

_dtype = tf.float64
_vars = []
_immediate = False
_finalized = False
_summary_writer = None

def _create_var(name: str, value_expr: tf.Tensor) -> None:
    """Internal helper for creating autosummary accumulators."""
    global _vars
    _vars.append(tf.Variable(initial_value=tf.zeros(shape=[], dtype=_dtype), name=name, trainable=False))
    update = tf.compat.v1.assign(_vars[-1], tf.cast(value_expr, _dtype))
    if _immediate:
        tf.compat.v1.get_default_session().run(update)
    return update

def autosummary(name: str, value, **kwargs) -> tf.Tensor:
    """Create a new autosummary."""
    if not hasattr(autosummary, "_tf_summary_step"):
        autosummary._tf_summary_step = tf.Variable(0, trainable=False, dtype=tf.int64)

    if _finalized:
        name_split = name.split("/")[-1]
        with tf.compat.v2.summary.record_if(True):
            tf.compat.v2.summary.scalar(name_split, value, step=autosummary._tf_summary_step)
            autosummary._tf_summary_step.assign_add(1)
        return value

    if kwargs:
        tf.compat.v1.summary.scalar(name, value, **kwargs)
    with tf.compat.v1.name_scope("Autosummary/" + name.replace("/", "_")):
        update = _create_var(name, value)
    if _immediate:
        value = tf.compat.v1.get_default_session().run(value)
    return value

def finalize_autosummaries() -> None:
    """Create summary ops for all pending autosummaries."""
    global _finalized
    _finalized = True

def save_summaries(log_dir: str):
    """Save all pending autosummaries."""
    if not _finalized:
        raise RuntimeError("save_summaries() called before finalize_autosummaries()")
        
    if not hasattr(save_summaries, "_summary_writer"):
        save_summaries._summary_writer = tf.compat.v2.summary.create_file_writer(log_dir)

    with save_summaries._summary_writer.as_default():
        for var in _vars:
            name = var.name.split(":")[0].replace("Autosummary/", "")
            value = var.value()
            with tf.compat.v2.summary.record_if(True):
                tf.compat.v2.summary.scalar(name, value, step=autosummary._tf_summary_step)
        autosummary._tf_summary_step.assign_add(1)
        save_summaries._summary_writer.flush()

def init_autosummary() -> None:
    """Initialize autosummary."""
    global _immediate
    _immediate = True

    global _vars
    _vars = []

    global _finalized
    _finalized = False
