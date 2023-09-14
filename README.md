# ACM Multimedia Systems 2024 Grand Challenge
## Offline Reinforcement Learning for Bandwidth Estimation in Real Time Communications 

## Grand Challenge Description

Video conferencing systems have recently emerged as indispensable tools to sustain global business operations and enable accessible education by revolutionizing the way people connect, collaborate, and communicate despite physical barriers and geographical divides. The quality of experience (QoE) delivered by these systems to the end user depends on bandwidth estimation, which is the problem of estimating the variable capacity of the bottleneck link between the sender and the receiver over time. In real time communication systems (RTC), the bandwidth estimate serves as a target bit rate for the audio/video encoder, controlling the send rate from the client. Overestimating the capacity of the bottleneck link causes network congestion as the client sends data at a rate higher than what the network can handle. Network congestion is characterized by increased delays in packet delivery, jitter, and potential packet losses. In terms of user’s experience, users will typically experience many resolution switches, frequent video freezes, garbled speech, and audio/video desynchronization, to name a few. Underestimating the available bandwidth on the other hand causes the client to encode and transmit the audio/video streams in a lower rate signal than what the network can handle, which leads to underutilization and degraded QoE. Estimating the available bandwidth accurately is therefore critical to providing the best possible QoE to users in RTC systems. Nonetheless, bandwidth estimation is faced with a multitude of challenges such as dynamic network paths between senders and receivers with fluctuating traffic loads, existence of diverse wired and wireless access network technologies with distinct characteristics, existence of different transmission protocols fighting for bandwidth to carry side and cross traffic, and partial observability of the network as only local packet statistics are available at the client side to base the estimate on.  

To improve QoE for users in RTC systems, this ACM MMSys 2024 grand challenge focuses on learning a fully data-driven bandwidth estimator using offline reinforcement learning based on a real-world dataset of packet traces with objective metrics that reflect user-perceived audio/video quality in Microsoft Teams. 

## Task

Offline reinforcement learning (RL) is a variant of RL where the agent learns from a fixed dataset of previously collected experiences, without interacting with the environment during training. In offline RL, the goal is to learn a policy that maximizes the expected cumulative reward based on the data. Offline RL is different from online RL where the agent can interact with the environment using its updated policy and learn from the feedback it receives online.  

In this challenge, participants are provided with a dataset of real-world packet traces for Microsoft Teams audio/video calls. Each packet trace corresponds to the sequence of packet headers received by the client in one audio/video call. In addition, objective signals which capture the user-perceived audio/video quality during the call are provided. This dataset is based on traces from real calls with different behaviour policies, including traditional and ML (machine learning) policies. The task of the challenge is to train a policy model (receiver side bandwidth estimator) which maps states (observed network statistics) to actions (bandwidth estimates) to improve QoE for users. To this end, participants are free to define the state-action spaces and reward functions, and process that data into a sequence of (states, actions, rewards) amenable for use with offline RL techniques, such as imitation learning, conservative Q-learning, inverse reinforcement learning, and constrained policy optimization. 


## Important Dates 

* Challenge announcement & website launch: September 30th, 2023 

* Dataset release: November 3rd, 2023 

* Model, code, and paper submission deadline: January 5th, 2024 

* Grand challenge paper acceptance: February 16th, 2024 

* Camera ready paper due: March 1st, 2024 

## Organizers

Sami Khairy, Gabriel Mittag, Ezra Ameri, Scott Inglis, Vishak Gopal, Mehrsa Golestaneh, Ross Cutler (Microsoft Corporation) 

Francis Yan, Zhixiong Niu (Microsoft Research) 

## Project

> This repo has been populated by an initial template to help get you started. Please
> make sure to update the content to build a great experience for community-building.

As the maintainer of this project, please make a few updates:

- Improving this README.MD file to provide a great experience
- Updating SUPPORT.MD with content about this project's support experience
- Understanding the security reporting process in SECURITY.MD
- Remove this section from the README

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

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
