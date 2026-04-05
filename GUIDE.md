# Founds
1. `V = max(R,G,B)` retains specular reflection peaks that ITU-R grayscale (`0.299R+0.587G+0.114B`) attenuates.
2. Targets was being linearly normalized, instead of std. Solved with StandardScaler, and applied the inverse transform at all evaluation points
3. Compared with ResNet50, based on literature for height measurement
4. More specific Feature Engineering based on analytical processing
	4.1 FFT
	4.3 Merge into multi-input with feature fusion (GENERATE IMAGE FOR THE ARCHITECTURE)
== Results ==
5. Necessary bias and gain calibration
6. LiDAR have VERY high RMSE (lots of 0s and very high outliers, making the filter unstable)
7. Metrics available in 'samples/section1/output/model_metrics.csv'

# Challenges
- Find the DC level
- How to explain each layer?
- Explain what is a 'Concatenate' layer
- What is the influence of each tabular data in the fusion models?
- Why the simplest model performs better?
- Grad-CAM will be used for future analysis with more data

# Obs
- Always resize feeding images using INTER_AREA
- Is needed to match the lowest and highest brightness level between the training samples and the real samples, because the lowest level in the synth samples are more greyish than the ones in the real dataset.
- The ground truth is not reliable, because the sensors are not ideal for this kind of application (radar is required for fluid level)
- What is the quantization used?
- Why is the summary saying 13mb model, if the .keras file is 41mb for WaterNet no fusion model?