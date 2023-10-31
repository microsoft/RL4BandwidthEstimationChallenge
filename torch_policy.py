import os
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
import torch
import torch.nn as nn

class MlBandwidthEstimator(nn.Module):
    def __init__(self, in_feat, hidden_size=256):
        super().__init__()
        self.in_feat = in_feat
        self.hidden_size = hidden_size
        # In this example, an lstm is used to construct a stateful deep net
        self.lstm = nn.LSTM(in_feat, hidden_size, num_layers=1, batch_first=True)
        # output layer: mean and standard deviation of bandwidth estimates
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x, h, c):
        h, c = (h.unsqueeze(0), c.unsqueeze(0)) # adding layer dimension
        self._features, [h, c] = self.lstm(x, [h, c])
        x = self.fc(self._features)
        h, c = (h.squeeze(0), c.squeeze(0)) # removing layer dimension
        return x, h, c  
    
if __name__ == "__main__":
    # batch size
    BS = 1
    # time steps
    T = 2000
    # observation vector dimension
    obs_dim = 150
    # number of hidden units in the LSTM
    hidden_size = 128
    
    # instantiate the ML BW estimator
    torchBwModel = MlBandwidthEstimator(in_feat=obs_dim, hidden_size=hidden_size)
    
    # create dummy inputs: 1 episode x T timesteps x obs_dim features
    dummy_inputs = np.asarray(np.random.uniform(0, 1, size=(BS, T, obs_dim)), dtype=np.float32)
    torch_dummy_inputs = torch.as_tensor(dummy_inputs)
    torch_initial_hidden_state = torch.zeros((BS, hidden_size))
    torch_initial_cell_state = torch.zeros((BS, hidden_size))

    # predict dummy outputs: 1 episode x T timesteps x 2 (mean and std)
    dummy_outputs, final_hidden_state, final_cell_state = torchBwModel(torch_dummy_inputs, torch_initial_hidden_state, torch_initial_cell_state)

    # save onnx model
    model_path = "./tmp/onnxBwModel.onnx"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torchBwModel.to("cpu")
    torchBwModel.eval()
    torch.onnx.export(
        torchBwModel,
        (torch_dummy_inputs, torch_initial_hidden_state, torch_initial_cell_state),
        model_path,
        opset_version=11,
        input_names=['obs', 'hidden_states', 'cell_states'], # the model's input names
        output_names=['output', 'state_out', 'cell_out'], # the model's output names
        dynamic_axes={
            'obs' : {0: 'batch_size', 1: 'seq_len'},
            'hidden_states' : {0: 'batch_size'},
            'cell_states' : {0: 'batch_size'},
            'state_out' : {0: 'batch_size'},
            'cell_out' : {0: 'batch_size'},                
            'output' : {0: 'batch_size', 1: 'seq_len'},
            }
    )
    
    # verify tf and onnx models outputs
    ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    onnx_hidden_state, onnx_cell_state = (np.zeros((1, hidden_size), dtype=np.float32), np.zeros((1, hidden_size), dtype=np.float32))
    torch_hidden_state, torch_cell_state = (torch.as_tensor(onnx_hidden_state), torch.as_tensor(onnx_cell_state))
    # online interaction: step through the environment 1 time step at a time
    with torch.no_grad():
        for i in tqdm(range(dummy_inputs.shape[1])):
            torch_estimate, torch_hidden_state, torch_cell_state = torchBwModel(torch_dummy_inputs[0:1, i:i+1, :], torch_hidden_state, torch_cell_state)
            feed_dict= {'obs': dummy_inputs[0:1,i:i+1,:], 'hidden_states': onnx_hidden_state, 'cell_states': onnx_cell_state}
            onnx_estimate, onnx_hidden_state, onnx_cell_state = ort_session.run(None, feed_dict)
            assert np.allclose(torch_estimate.numpy(), onnx_estimate, atol=1e-6), 'Failed to match model outputs!'
            assert np.allclose(torch_hidden_state, onnx_hidden_state, atol=1e-7), 'Failed to match hidden state1'
            assert np.allclose(torch_cell_state, onnx_cell_state, atol=1e-7), 'Failed to match cell state!'
        
        assert np.allclose(torch_hidden_state, final_hidden_state, atol=1e-7), 'Failed to match final hidden state!'
        assert np.allclose(torch_cell_state, final_cell_state, atol=1e-7), 'Failed to match final cell state!'
        print("Torch and Onnx models outputs have been verified successfully!")