---
license: cc-by-4.0
---
## Dataset Description:

This is a RL dataset for general function-calling by utilizing existing expert tool-use trajectories. We pose each assistant step of the trajectory as a separate behavior cloning problem where the policy model is incentivized to match the tool call choices of the expert model.

This dataset is released as part of NVIDIA [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym), a framework for building reinforcement learning environments to train large language models. NeMo Gym contains a growing collection of training environments and datasets to enable Reinforcement Learning from Verifiable Reward (RLVR). This dataset was utilized in the development of the [NVIDIA Nemotron](https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/) family of models.

NeMo Gym is an open-source library within the [NVIDIA NeMo framework](https://github.com/NVIDIA-NeMo/), NVIDIA's GPU accelerated, end-to-end training framework for large language models (LLMs), multi-modal models and speech models.

This dataset is part of the https://huggingface.co/collections/nvidia/nemo-gym/ collection

This dataset is ready for commercial use.

## Dataset Owner(s):
NVIDIA Corporation

## Dataset Creation Date:
03/11/2026

## License/Terms of Use: 
This dataset is licensed under Creative Commons Attribution 4.0 International (CC-BY 4.0). Additional Information: Apache 2.0 License; MIT License.

## Intended Usage:
To be used with [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) for post-training LLMs. 

## Dataset Characterization
* Data Collection Method<br>
  * [Synthetic] <br>
Using deepseek-ai/DeepSeek-V3.2, zai-org/GLM-4.6, openai/gpt-oss-120b, moonshotai/Kimi-K2-Instruct

* Labeling Method<br>
  * Not Applicable 

## Dataset Format
Structured conversations in JSON, Compatible with https://github.com/NVIDIA-NeMo/Gym

## Dataset Quantification

Record Count: Train 8458: 

Feature Count: 5 main fields per record (trajectory_id, info, responses_create_params, expected_action, agent_ref)

Total Data Storage: Train 389MB, Val 9.5MB

## Reference(s):
[NeMo Gym](https://github.com/NVIDIA-NeMo/Gym)

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal teams to ensure this dataset meets requirements for the relevant industry and use case and addresses unforeseen product misuse.   
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).  