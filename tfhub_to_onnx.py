import tensorflow_hub as hub
import tensorflow as tf
import onnxmltools

# tf.compat.v1.disable_eager_execution()
# sess = tf.compat.v1.Session()
IMAGE_SIZE = (224,224)
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer("https://tfhub.dev/google/remote_sensing/bigearthnet-resnet50/1",tags='train'),
    ])
model.build((None,)+IMAGE_SIZE+(3,))

# model.save('run1')
onnx_model = onnxmltools.convert_keras(model, target_opset=6)
keras2onnx.save_model(onnx_model, 'model.onnx')
