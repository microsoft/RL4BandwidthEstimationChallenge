# ACM Multimedia Systems 2024 Grand Challenge
## Offline Reinforcement Learning for Bandwidth Estimation in Real Time Communications 

## Challenge Website

The complete details about the challenge and participation rules can be found on the [challenge website](https://www.microsoft.com/en-us/research/academic-program/bandwidth-estimation-challenge/). To register for the challenge, please fill out this [form](https://forms.office.com/r/LE5PR9EL5q). 

## Announcements 

* [October 20th, 2023]: [testbed dataset](https://github.com/microsoft/RL4BandwidthEstimationChallenge/blob/main/download-testbed-dataset.sh) and [emulated dataset](https://github.com/microsoft/RL4BandwidthEstimationChallenge/blob/main/download-emulated-dataset.sh) are released.
* [November 24th, 2023]: [baseline](https://github.com/microsoft/RL4BandwidthEstimationChallenge/tree/main/onnx_models) offline RL model trained on the emulated dataset is released.
* [November 24th, 2023]: the details and instructions for the preliminary evaluation opportunity are posted on the [challenge website](https://www.microsoft.com/en-us/research/academic-program/bandwidth-estimation-challenge/preliminary-evaluation/). 
* [December 6th, 2023]: preliminary evaluation submission deadline extended to December 8th, 2023 11:59 PM AoE. torch_policy.py [bug fixed](https://github.com/microsoft/RL4BandwidthEstimationChallenge/commit/7258bda0e87d2f7f1ad8a8380806e45bb88c29d7).

## Grand Challenge Description

Video conferencing systems have recently emerged as indispensable tools to sustain global business operations and enable accessible education by revolutionizing the way people connect, collaborate, and communicate despite physical barriers and geographical divides. The quality of experience (QoE) delivered by these systems to the end user depends on bandwidth estimation, which is the problem of estimating the variable capacity of the bottleneck link between the sender and the receiver over time. In real time communication systems (RTC), the bandwidth estimate serves as a target bit rate for the audio/video encoder, controlling the send rate from the client. Overestimating the capacity of the bottleneck link causes network congestion as the client sends data at a rate higher than what the network can handle. Network congestion is characterized by increased delays in packet delivery, jitter, and potential packet losses. In terms of user’s experience, users will typically experience many resolution switches, frequent video freezes, garbled speech, and audio/video desynchronization, to name a few. Underestimating the available bandwidth on the other hand causes the client to encode and transmit the audio/video streams in a lower rate signal than what the network can handle, which leads to underutilization and degraded QoE. Estimating the available bandwidth accurately is therefore critical to providing the best possible QoE to users in RTC systems. Nonetheless, bandwidth estimation is faced with a multitude of challenges such as dynamic network paths between senders and receivers with fluctuating traffic loads, existence of diverse wired and wireless access network technologies with distinct characteristics, existence of different transmission protocols fighting for bandwidth to carry side and cross traffic, and partial observability of the network as only local packet statistics are available at the client side to base the estimate on.

To improve QoE for users in RTC systems, the ACM MMSys 2024 grand challenge focuses on the development of a deep learning-based bandwidth estimator using offline reinforcement learning (RL) techniques. A real-world dataset of observed network dynamics with objective metrics that reflect user-perceived audio/video quality in Microsoft Teams is released to train the deep RL policies for bandwidth estimation.

**Please NOTE** that the intellectual property (IP) is not transferred to the challenge organizers, i.e., participants remain the owners of their code (when the code is made publicly available, an appropriate license should be added).

## Challenge Task

Offline RL is a variant of RL where the agent learns from a fixed dataset of previously collected experiences, without interacting with the environment during training. In offline RL, the goal is to learn a policy that maximizes the expected cumulative reward based on the dataset. Offline RL is different from online RL where the agent can interact with the environment using its updated policy and learn from the feedback it receives online.   

In this challenge, participants are provided with a dataset of real-world trajectories for Microsoft Teams audio/video calls. Each trajectory corresponds to the sequence of high-dimensional observation vector (on) computed based on packet information received by the client in one audio/video call, along with the bandwidth estimates (bn) predicted by different estimators (behavior policies). Objective signals which capture the user-perceived audio/video quality during the call are provided. These objective signals are predicted by ML models whose predictions have high correlation with subjective audio and video quality scores as determined by ITU-T's P.808 and P.910, respectively.

The goal of the challenge is to improve QoE for RTC system users as measured by objective audio/video quality scores by developing a deep learning-based policy model (receiver-side bandwidth estimator, π) with offline RL techniques, such as conservative Q-learning, inverse reinforcement learning, and constrained policy optimization, to name a few. To this end, participants are free to specify an appropriate reward function based on the provided dataset of observed network dynamics and objective metrics, the model architecture, and the training algorithm, given that the developed model adheres to the below requirements.

## Challenge Requirements

1. The policy model ($\pi$) can be a state-less or a stateful model that outputs the bandwidth estimate ($b_n$) in bits per second (bps). The input to a stateless model is the observation vector ($o_n$), hence, $\pi_\text{stateless}: o_n \rightarrow b_n$. On the other hand, the inputs to a stateful model are the observation vector ($o_n$), as well as hidden ($h_{n-1}$) and cell ($c_{n-1}$)  states which are representations learned by the model to capture the underlying structure and temporal dependencies in the sequence of observation vectors, hence, $\pi_\text{stateful}: o_n, h_{n-1}, c_{n-1} \rightarrow b_n$. Please refer to the [TF model class](https://github.com/microsoft/RL4BandwidthEstimationChallenge/blob/main/tf_policy.py) or [PyTorch model class](https://github.com/microsoft/RL4BandwidthEstimationChallenge/blob/main/torch_policy.py) in the repository which shows the required inputs and outputs. Any policy model that does not adhere to this input/output signature will be disqualified from the competition.

2. Feature transformation and/or feature selection should be performed in a processing block within the model. For instance, the first layer ($l_0$) of the model can map the observation vector ($o_n$) to a desired agent state ($s_n$), $l_0: o_n → s_n$.

3. Participants can specify an appropriate action space, e.g. $a_n \in [0,1]$, however, the transformation from the action space to the bps space should be performed by the last layer ($l_N$) of the model such that the model predicts the bandwidth estimates in bps, $l_N: a_n \rightarrow b_n$.

4. Participants can specify an appropriate reward function for training the RL agent based on the provided signals: audio quality signal, video quality signal, and network metrics in the observation vector.

5. To reduce the hardware requirements when the policy model is used for inference at the client side of the video conferencing system, the model size must be smaller than 10 MB and inference latency should be no more than 5ms on an Intel Core i5 Quadcore clocked at 2.4 GHz using a single thread. 

6. In offline RL it is typical to use an actor-critic architecture. As long as the inputs to the actor/policy model adhere to the aforementioned requirements, any set of features can be used as inputs for the critic.

7. Participants can train the model using PyTorch or TensorFlow, and the model should be exported to ONNX. To ensure that organizers can run the model correctly, participants are required to share a small subset of their validation data along with their model outputs to be used for verification. We provide sample scripts to convert [PyTorch](https://github.com/microsoft/RL4BandwidthEstimationChallenge/blob/main/torch_policy.py) and [TF models](https://github.com/microsoft/RL4BandwidthEstimationChallenge/blob/main/tf_policy.py) in the repository. We have also released a [baseline stateless model (MLP)](https://github.com/microsoft/RL4BandwidthEstimationChallenge/blob/main/onnx_models/) as a reference, with an [example script](https://github.com/microsoft/RL4BandwidthEstimationChallenge/blob/main/run_baseline_model.py) to run this model.

8. Participants should submit their training code to the Open-source Software and Datasets track of the conference to receive a reproducibility badge.

## Dataset Description

Please refer to the [dataset description](https://www.microsoft.com/en-us/research/academic-program/bandwidth-estimation-challenge/data/) on the challenge website.

## In this Repository

This repository contains scripts required for 2nd Bandwidth Estimation Challenge at ACM MMSys 2024. 

1. [Script to download the Testbed dataset](https://github.com/microsoft/RL4BandwidthEstimationChallenge/blob/main/download-testbed-dataset.sh)

2. [Script to download the Emulated dataset](https://github.com/microsoft/RL4BandwidthEstimationChallenge/blob/main/download-emulated-dataset.sh)

3. [Bandwidth estimator model class in Tensorflow and converison to onnx](https://github.com/microsoft/RL4BandwidthEstimationChallenge/blob/main/tf_policy.py)

4. [Bandwidth estimator model class in PyTorch and converison to onnx](https://github.com/microsoft/RL4BandwidthEstimationChallenge/blob/main/torch_policy.py)

5. [Offline RL baseline ONNX model](https://github.com/microsoft/RL4BandwidthEstimationChallenge/tree/main/onnx_models) and [inference script](https://github.com/microsoft/RL4BandwidthEstimationChallenge/blob/main/run_baseline_model.py)

6. [Code prerequisites](https://github.com/microsoft/RL4BandwidthEstimationChallenge/blob/main/requirements.txt)

## Important Dates 

* Challenge announcement & website launch: October 9th, 2023 

* Dataset release: October 20th, 2023 

* Model, code, and paper submission deadline: January 5th, 2024 

* Grand challenge paper acceptance: February 16th, 2024 

* Camera ready paper due: March 1st, 2024 

## Citation

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Legal Notices

Microsoft and any contributors grant you a license to the Microsoft documentation and other content
in this repository under the [Creative Commons Attribution 4.0 International Public License](https://creativecommons.org/licenses/by/4.0/legalcode),
see the [LICENSE](LICENSE) file, and grant you a license to any code in the repository under the [MIT License](https://opensource.org/licenses/MIT), see the
[LICENSE-CODE](LICENSE-CODE) file.

Microsoft, Windows, Microsoft Azure and/or other Microsoft products and services referenced in the
documentation may be either trademarks or registered trademarks of Microsoft in the United States
and/or other countries. The licenses for this project do not grant you rights to use any Microsoft
names, logos, or trademarks. Microsoft's general trademark guidelines can be found at
http://go.microsoft.com/fwlink/?LinkID=254653.

Privacy information can be found at https://privacy.microsoft.com/en-us/

Microsoft and any contributors reserve all other rights, whether under their respective copyrights, patents,
or trademarks, whether by implication, estoppel or otherwise.

## Dataset licenses

MICROSOFT PROVIDES THE DATASETS ON AN "AS IS" BASIS. MICROSOFT MAKES NO WARRANTIES, EXPRESS OR IMPLIED, GUARANTEES OR CONDITIONS WITH RESPECT TO YOUR USE OF THE DATASETS. TO THE EXTENT PERMITTED UNDER YOUR LOCAL LAW, MICROSOFT DISCLAIMS ALL LIABILITY FOR ANY DAMAGES OR LOSSES, INLCUDING DIRECT, CONSEQUENTIAL, SPECIAL, INDIRECT, INCIDENTAL OR PUNITIVE, RESULTING FROM YOUR USE OF THE DATASETS.

## Code license
MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE

## Organizers

Sami Khairy, Gabriel Mittag, Ezra Ameri, Scott Inglis, Vishak Gopal, Mehrsa Golestaneh, Ross Cutler (Microsoft Corporation) 

Francis Yan, Zhixiong Niu (Microsoft Research) 