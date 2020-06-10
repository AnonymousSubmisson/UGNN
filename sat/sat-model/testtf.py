import tensorflow as tf
import numpy as np

def datagen():
    while True:
        print('datagen')
        yield np.array([1., 2., 3.], dtype=np.float32), 

d = tf.data.Dataset.from_generator(datagen, (tf.float32,), (tf.TensorShape([3]),))
d_it = d.make_one_shot_iterator()
d_val,  = d_it.get_next()

ph = tf.placeholder(dtype=tf.float32, shape=[3])
r = tf.math.reduce_sum(ph)

sess = tf.Session()
print('sess')
print(sess.run(r, {ph: d_val}))
print('1')
print(sess.run(r, {ph: d_val}))
print('2')
