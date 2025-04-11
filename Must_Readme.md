Summary of Alignment with Mid-Air
Coordinates:
All modalities (RGB, depth, IMU, GPS, ground-truth, predicted poses) are in NED, either natively or via transformations (to_ned_pose, IMU axis mapping).
Status: Fully aligned.
Absolute/Relative Conversions:
Absolute to relative (ground-truth): Correctly computed in NED.
Relative to absolute (predicted): Correctly integrated in NED.
Status: Fully aligned.
Frequency Alignment and Timestamping:
All data aligned at 25 Hz (visual data rate) via subsampling.
Timestamping implicit via frame indices, consistent with Mid-Air’s synchronized data.
Status: Fully aligned.
Loss Computation:
Relative pose loss: Aligned (NED, same scale).
GPS loss: Aligned after denormalization (NED, meters).
Depth loss: Aligned after denormalization (NED, meters).
Status: Aligned with the fix above.
Conclusion
Your pipeline now fully adheres to Mid-Air’s technical specifications. The NED convention is consistently applied, conversions are correct, frequencies are aligned, and loss computations are on the same page after the normalization fix. You’re good to continue training! Monitor wandb logs to ensure depth_loss and gps_loss behave as expected after denormalization. Let me know if you need further adjustments!