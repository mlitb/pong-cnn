import tensorflow as tf
tf.enable_eager_execution()
tfe = tf.contrib.eager
n = 10
x = tf.random_normal([n, 2])
noise = tf.random_normal([n, 2])
y = x * 3 + 2 + noise

class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    self.W = tfe.Variable(5., name='weight')
    self.B = tfe.Variable(10., name='bias')
  # Overriding call not predict
  def call(self, inputs):
    return inputs * self.W + self.B

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

model = Model()
with tf.GradientTape() as tape:
  error = model(x) - y
  print(error.shape)
  loss_value = tf.reduce_mean(tf.square(error))
gradients = tape.gradient(loss_value, model.variables)
print(gradients)
optimizer.apply_gradients(zip(gradients, model.variables),
                            global_step=tf.train.get_or_create_global_step())