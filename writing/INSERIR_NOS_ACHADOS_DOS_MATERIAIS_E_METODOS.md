# Founds
1. `V = max(R,G,B)` retains specular reflection peaks that ITU-R grayscale (`0.299R+0.587G+0.114B`) attenuates.
2. Targets was being linearly normalized, instead of std. Solved with StandardScaler, and applied the inverse transform at all evaluation points
3. ResNet50, based on literature
4. Redone data pipeline
5. More specific Feature Engineering based on analytical processing
	5.1 FFT
	5.2 Gradient Detection for Edges (Sobel based)
	5.3 Merge into multi-input with feature fusion (GENERATE IMAGE FOR THE ARCHITECTURE)
6. NOT NECESSARY FURTHER DATA AUGMENTATION, check the random gamma for sun simulation
== Results ==
7. Bias calibration
8. Future analysis with Grad-CAM

# Challenges
- Find the DC level
- How to explain each layer?
- Explain what is a 'Concatenate' layer
- What is the influence of each tabular data in the fusion models?
- Why the simplest model performs better?

# Obs
- Always resize images using INTER_AREA
- Match the lowest and highest level between the training samples and the real samples, because the lowest level in the synth samples are more greyish than the ones in the real dataset. Adjust brightness function
- The ground truth is not reliable, because the sensors are not ideal for this kind of application (radar is required for fluid level)
- What is the quantization used?
- Why is the summary saying 13mb model, if the .keras file is 41mb?