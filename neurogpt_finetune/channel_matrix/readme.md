NeuroGPT is restricted to 22 channels, so each downstream task must have a channel-mapping matrix (.npy).
To compute this matrix, 
(1) Prepare the downstream task’s channel info in the exact input order
(2) update the "# your channel in order" section in matrix_mne.iypnb with these channel info

The resulting .npy file is the required conversion matrix for NeuroGPT.
