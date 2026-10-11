import tensorflow as tf

info = tf.sysconfig.get_build_info()
print("TensorFlow    :", tf.__version__)
print("built for CUDA:", info.get("cuda_version"))
print("built for cuDNN:", info.get("cudnn_version"))

gpus = tf.config.list_physical_devices("GPU")
print("GPUs          :", gpus)
if not gpus:
    raise SystemExit(1)

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

with tf.device("/GPU:0"):
    a = tf.random.normal((2048, 2048))
    b = tf.random.normal((2048, 2048))
    c = tf.matmul(a, b)
print("matmul device :", c.device)