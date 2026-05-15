import joblib
import numpy as np
import pandas as pd

class model:
    def __init__(self):
        self.model = joblib.load('model/isolation_forest_model.pkl')
        self.scaler = joblib.load('model/scaler.pkl')

    def extract_feature(self,test_df, window=50):
        test_df["vibration"] = np.sqrt(test_df["ax"] ** 2 + test_df["ay"] ** 2 + test_df["az"] ** 2)
        test_df["trend"] = test_df["vibration"].rolling(window=window, center=True, min_periods=1).mean()
        test_df["vibration"] = test_df["vibration"] - test_df["trend"]

        test_df["magnitude"] = np.sqrt(test_df["gx"] ** 2 + test_df["gy"] ** 2 + test_df["gz"] ** 2)

        test_df_feats = test_df.copy()[["vibration", "magnitude"]]

        features = []

        for i in range(0, len(test_df_feats) - window, window):
            vib = test_df["vibration"][i:i + window]
            mag = test_df["magnitude"][i:i + window]

            features.append({
                "vib_mean": np.mean(vib),
                "vib_std": np.std(vib),
                "vib_rms": np.sqrt(np.mean(vib ** 2)),  # <- this is RMS, not kurtosis
                "vib_max": np.max(vib),

                "mag_mean": np.mean(mag),
                "mag_std": np.std(mag),
                "mag_rms": np.sqrt(np.mean(mag ** 2)),
                "mag_max": np.max(mag),
            })

        return pd.DataFrame(features)
