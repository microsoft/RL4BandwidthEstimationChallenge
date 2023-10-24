import onnx
import tf2onnx
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import onnxruntime as ort

class MlBandwidthEstimator(tf.keras.Model):
    def __init__(self, stateful=True, lstm_units = 128, hidden_units=(128,128,128)):
        super(MlBandwidthEstimator, self).__init__()
        self.stateful = stateful
        self.model_layers = []
        if self.stateful:
            # In this example, an lstm is used to construct a stateful deep net.
            # Note that any memory cell can be used given that the model input signature remains the same.
            self.model_layers.append(tf.keras.layers.Dense(units=h, activation=tf.nn.tanh, kernel_initializer=tf.keras.initializers.Orthogonal(gain=2.0**0.5), bias_initializer='zeros'))
        for h in hidden_units:
            self.model_layers.append(tf.keras.layers.Dense(units=h, activation=tf.nn.tanh))
        # output layer: mean and standard deviation of bandwidth estimates
        self.model_layers.append(tf.keras.layers.Dense(units=2, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.Orthogonal(gain=2.0**0.5), bias_initializer='zeros'))
    
    def call(self, inputs):
        # inputs to the model is a tuple of (observation_vector, hidden_state, cell_state)
        observation_vector, hidden_state, cell_state = inputs[0], inputs[1], inputs[2]
        if self.stateful:
            lstm_state = (hidden_state, cell_state)
            # For a GRU cell: lstm_state = hidden_state
            encoding, next_hidden_state, next_cell_state = self.lstm_layer(observation_vector, initial_state=lstm_state)
        else:
            # For a stateless model, next_hidden_state, next_cell_state are the same as input hidden_state and cell_state, respectively.
            next_hidden_state, next_cell_state = hidden_state, cell_state
            encoding = observation_vector
        for layer in self.model_layers:
            encoding = layer(encoding)
        return encoding, next_hidden_state, next_cell_state
    

if __name__ == "__main__":
    # time steps
    T = 2000
    # observation vector dimension
    obs_dim = 150
    # number of hidden units in the LSTM
    lstm_units = 128
    
    # instantiate the ML BW estimator
    tfBwModel = MlBandwidthEstimator(stateful=False, lstm_units = lstm_units, hidden_units=(128,128,128,128,128))
    
    # create dummy inputs: 1 episode x T timesteps x obs_dim features
    dummy_inputs = np.asarray(np.random.uniform(0, 1, size=(1,T,obs_dim)), dtype=np.float32)
    initial_hidden_state, initial_cell_state = (tf.zeros((1, lstm_units), dtype=tf.float32), tf.zeros((1, lstm_units), dtype=tf.float32))
    # predict dummy outputs: 1 episode x T timesteps x 2 (mean and std)
    dummy_outputs, final_hidden_state, final_cell_state = tfBwModel((dummy_inputs, initial_hidden_state, initial_cell_state))

    # save tf model
    saved_model_dir = "./tmp/tf_model"
    tfBwModel.save(saved_model_dir)

    # convert tf model to onnx
    input_signature = [tf.TensorSpec([1, 1, obs_dim], tf.float32, name='obs'), tf.TensorSpec([1,lstm_units], tf.float32, name='hidden_states'), tf.TensorSpec([1,lstm_units], tf.float32, name='cell_states')]
    onnxBwModel, _ = tf2onnx.convert.from_keras(tfBwModel, input_signature=input_signature, opset=11)
    
    # save onnx model
    model_path = "./tmp/onnxBwModel.onnx"
    onnx.save(onnxBwModel, model_path)
    
    # verify tf and onnx models outputs
    ort_session = ort.InferenceSession(onnxBwModel.SerializeToString())
    onnx_hidden_state, onnx_cell_state = (np.zeros((1, lstm_units),dtype=np.float32), np.zeros((1, lstm_units),dtype=np.float32))
    tf_hidden_state, tf_cell_state = (tf.zeros((1, lstm_units), dtype=tf.float32), tf.zeros((1, lstm_units), dtype=tf.float32))
    # online interaction: step through the environment 1 time step at a time
    for i in tqdm(range(dummy_inputs.shape[1])):
        tf_estimate, tf_hidden_state, tf_cell_state = tfBwModel((dummy_inputs[0:1,i:i+1,:],tf_hidden_state,tf_cell_state))
        feed_dict= {'obs':dummy_inputs[0:1,i:i+1,:],'hidden_states':onnx_hidden_state,'cell_states':onnx_cell_state}
        onnx_estimate, onnx_hidden_state, onnx_cell_state = ort_session.run(None, feed_dict)
        assert np.allclose(tf_estimate,onnx_estimate,atol=1e-6), 'Failed to match model outputs!'
        assert np.allclose(tf_hidden_state,onnx_hidden_state,atol=1e-8), 'Failed to match hidden state1'
        assert np.allclose(tf_cell_state,onnx_cell_state,atol=1e-8), 'Failed to match cell state!'
    
    assert np.allclose(tf_hidden_state,final_hidden_state,atol=1e-8), 'Failed to match final hidden state!'
    assert np.allclose(tf_cell_state,final_cell_state,atol=1e-8), 'Failed to match final cell state!'
    print("TF and Onnx models outputs have been verified successfully!")