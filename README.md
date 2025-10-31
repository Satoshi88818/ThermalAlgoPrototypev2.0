# ThermalAlgoPrototypev2.0

Overview
ThermalAlgoPrototypev2.0 is a production-ready, first-principles-engineered Discrete Diffusion Transformer Model (DTM) optimized for 64×64 RGB image generation using bit-flip forward processes, binary score matching, and Persistent Contrastive Divergence (PCD-k). It leverages block-sparse energy functions and supports TSU (Thermal Sampling Unit) hardware acceleration with seamless GPU/CPU fallback.
This model is designed for high-fidelity generative modeling on binarized image data (e.g., CIFAR-10), achieving FID-competitive performance through efficient block-wise Gibbs sampling and sparse bilinear interactions.











































FeatureDescription64×64 RGB GenerationFull-color, high-resolution discrete image synthesisBit-Flip Forward ProcessDeterministic noise schedule via bitflip_schedule()Binary DSM + PCD-k TrainingCombines score matching and contrastive divergenceBlock-Sparse Energy Function90%+ sparsity in local/skip connectionsTSU-Native (Optional)Hardware acceleration via thrml libraryGPU/CPU FallbackFull functionality without TSUFID Evaluation ReadyBuilt-in FID computation using torch-fidCheckpointing & SamplingAuto-save + grid visualization
