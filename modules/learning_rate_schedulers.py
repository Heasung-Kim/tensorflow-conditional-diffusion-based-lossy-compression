import tensorflow as tf

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, scheduler_function):
        self.initial_lr = initial_lr
        self.scheduler_function = scheduler_function

    def __call__(self, step):
        return self.scheduler_function(self.initial_lr, step)


