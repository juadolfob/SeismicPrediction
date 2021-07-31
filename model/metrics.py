import sklearn
import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow import keras


class NegativePredictiveValue(tf.keras.metrics.Metric):

    def __init__(self, name='negativ_predictive_value', **kwargs):
        super(NegativePredictiveValue, self).__init__(name=name, **kwargs)
        self.true_negatives = self.add_weight('true_negatives', initializer=init_ops.zeros_initializer)
        self.false_negatives = self.add_weight('false_negatives', initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(tf.round(y_pred), tf.bool)
        # if (y_pred == negative_value) and (y_pred == negative_value)
        self.true_negatives.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_pred, False),
                                                                            tf.equal(y_pred, y_true)), tf.float32)))
        self.false_negatives.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_pred, False),
                                                                             tf.not_equal(y_pred, y_true)),
                                                              tf.float32)))

    def result(self):
        return self.true_negatives / (self.true_negatives + self.false_negatives + tf.keras.backend.epsilon())

    def reset_state(self):
        self.true_negatives.assign(0)
        self.false_negatives.assign(0)


class RScore(tf.keras.metrics.Metric):

    def __init__(self, name='r_score', **kwargs):
        super(RScore, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight('true_positives', initializer=init_ops.zeros_initializer)
        self.true_negatives = self.add_weight('true_negatives', initializer=init_ops.zeros_initializer)
        self.false_positives = self.add_weight('false_positives', initializer=init_ops.zeros_initializer)
        self.false_negatives = self.add_weight('false_negatives', initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(tf.round(y_pred), tf.bool)
        self.true_positives.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_pred, True),
                                                                            tf.equal(y_pred, y_true)), tf.float32)))
        self.true_negatives.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_pred, False),
                                                                            tf.equal(y_pred, y_true)), tf.float32)))
        self.false_positives.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_pred, True),
                                                                             tf.not_equal(y_pred, y_true)),
                                                              tf.float32)))
        self.false_negatives.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_pred, False),
                                                                             tf.not_equal(y_pred, y_true)),
                                                              tf.float32)))

    def result(self):
        return ((self.true_positives * self.true_negatives) - (self.false_positives * self.false_negatives)) / (
                    (self.true_positives + self.false_negatives) * (self.false_positives + self.true_negatives))

    def reset_state(self):
        self.true_positives.assign(0)
        self.true_negatives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)


class MCC(tf.keras.metrics.Metric):

    def __init__(self, name='mcc', **kwargs):
        super(MCC, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight('true_positives', initializer=init_ops.zeros_initializer)
        self.true_negatives = self.add_weight('true_negatives', initializer=init_ops.zeros_initializer)
        self.false_positives = self.add_weight('false_positives', initializer=init_ops.zeros_initializer)
        self.false_negatives = self.add_weight('false_negatives', initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(tf.round(y_pred), tf.bool)
        self.true_positives.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_pred, True),
                                                                            tf.equal(y_pred, y_true)), tf.float32)))
        self.true_negatives.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_pred, False),
                                                                            tf.equal(y_pred, y_true)), tf.float32)))
        self.false_positives.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_pred, True),
                                                                             tf.not_equal(y_pred, y_true)),
                                                              tf.float32)))
        self.false_negatives.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_pred, False),
                                                                             tf.not_equal(y_pred, y_true)),
                                                              tf.float32)))

    def result(self):
        return ((self.true_positives * self.true_negatives) - (self.false_positives * self.false_negatives)) / tf.sqrt((
                                                                                                                                   (
                                                                                                                                               self.true_positives + self.false_positives) * (
                                                                                                                                               self.true_positives + self.false_negatives) * (
                                                                                                                                               self.true_negatives + self.false_positives) * (
                                                                                                                                               self.true_negatives + self.false_negatives)))

    def reset_state(self):
        self.true_positives.assign(0)
        self.true_negatives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)


class FalsePositiveRate(tf.keras.metrics.Metric):

    def __init__(self, name='false_positive_rate', **kwargs):
        super(FalsePositiveRate, self).__init__(name=name, **kwargs)
        self.true_negatives = self.add_weight('true_negatives', initializer=init_ops.zeros_initializer)
        self.false_positives = self.add_weight('false_positives', initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(tf.round(y_pred), tf.bool)
        self.true_negatives.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_pred, False),
                                                                            tf.equal(y_pred, y_true)), tf.float32)))
        self.false_positives.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_pred, True),
                                                                             tf.not_equal(y_pred, y_true)),
                                                              tf.float32)))

    def result(self):
        return self.true_negatives / (self.true_negatives + self.false_positives)

    def reset_state(self):
        self.true_negatives.assign(0)
        self.false_positives.assign(0)


METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='Accuracy'),
    keras.metrics.Precision(name='Precision'),
    keras.metrics.Recall(name='Recall'),
    sklearn.metrics.f1_score,
    NegativePredictiveValue(name='NPV'),
    RScore(name='R score'),
    MCC(name='MCC'),
    FalsePositiveRate(name='FPR'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'),
]
