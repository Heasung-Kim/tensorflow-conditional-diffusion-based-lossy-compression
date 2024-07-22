import tensorflow as tf
from tensorflow.data import Dataset
from .load_dataset import load_dataset
# from .transposed_collate import train_transposed_collate, test_transposed_collate


def load_data(data_config, batch_size, num_workers=4, sequence=False):
    """
    Wrapper around load_dataset. Gets the dataset, then places it in a tf.data.Dataset.

    Args:
        data_config (dict): data configuration dictionary
        batch_size (int): batch size for training and validation
        num_workers (int): number of threads of multi-processed data loading
        sequence (bool): whether data examples are sequences, in which case the
                         data loader returns transposed batches with the sequence
                         step as the first dimension and batch index as the
                         second dimension
    """
    train, val = load_dataset(data_config)

    if train is not None:
        if sequence:
            train = train.map(train_transposed_collate, num_parallel_calls=num_workers)
        #train = train.cache().batch(batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)
        train = train.batch(batch_size).cache().repeat().prefetch(tf.data.experimental.AUTOTUNE)

    if val is not None:

        if sequence:
            val = val.map(test_transposed_collate, num_parallel_calls=num_workers)
        #val = val.cache().batch(batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)
        val = val.batch(batch_size).cache().repeat().prefetch(tf.data.experimental.AUTOTUNE)
        #val = val.batch(batch_size)#.repeat()

    return train, val