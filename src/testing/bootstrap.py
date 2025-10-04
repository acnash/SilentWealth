import pandas as pd
import numpy as np


class Bootstrap:

    def __init__(self, data_file, bootstrap_sample_size, bootstrap_number_samples):
        self.df = pd.read_csv(data_file)
        self.bootstrap_sample_size = bootstrap_sample_size
        self.bootstrap_number_samples = bootstrap_number_samples
        self.bootstrap_samples = []

    def build_samples(self):
        max_start_index = len(self.df) - self.bootstrap_sample_size
        for _ in range(self.bootstrap_number_samples):
            start_index = np.random.randint(0, max_start_index + 1)
            sample_df = self.df.iloc[start_index:start_index + 1440].reset_index(drop=True)
            self.bootstrap_samples.append(sample_df)

        return self.bootstrap_samples
