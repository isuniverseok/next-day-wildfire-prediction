import numpy as np
import torch

# fmt: off
INPUT_VARS = [
    "elevation", "th", "vs", "tmmn", "tmmx", "sph", "pr",
    "pdsi", "NDVI", "population", "erc", "PrevFireMask",
]
OUTPUT_VARS = ["FireMask"]
# fmt: on


class WildfireDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_file,
        input_vars=INPUT_VARS,
        output_vars=OUTPUT_VARS,
        transform=None,
        target_transform=None,
        transforms=None,
    ):
        self.data = np.load(data_file, allow_pickle=True)
        self.data = {key: self.data[key] for key in self.data.keys()}
        self.data_stats = self.data.get("data_stats")
        all_vars = INPUT_VARS + OUTPUT_VARS
        assert set(all_vars).issubset(
            set(self.data.keys())
        ), f"Data file missing vars: {all_vars}"

        self.input_vars = input_vars
        self.output_vars = output_vars
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms

    def __len__(self):
        return len(self.data[self.input_vars[0]])

    def __getitem__(self, idx):
        
        input_data = np.stack([self.data[var][idx] for var in self.input_vars], axis=0)
        target_data = np.stack([self.data[v][idx] for v in self.output_vars], axis=0)

        
        for i, var in enumerate(self.input_vars): # Normalize the input data (as discussed in lecture slides)
            mean = DATA_STATS[var][2]  # Mean
            std = DATA_STATS[var][3]  # Standard deviation
            input_data[i] = (input_data[i] - mean) / std

        input_data = torch.tensor(input_data, dtype=torch.float32)
        target_data = torch.tensor(target_data, dtype=torch.float32)

        if self.transform:
            input_data = self.transform(input_data)
        if self.target_transform:
            target_data = self.target_transform(target_data)
        if self.transforms:
            input_data, target_data = self.transforms(input_data, target_data)
        return input_data, target_data

DATA_STATS = {
    # Elevation in m.
    # 0.1 percentile, 99.9 percentile
    "elevation": (0.0, 3141.0, 657.3003, 649.0147),
    # Drought Index (Palmer Drought Severity Index)
    # 0.1 percentile, 99.9 percentile
    "pdsi": (-6.12974870967865, 7.876040384292651, -0.0052714925, 2.6823447),
    # Vegetation index (times 10,000 maybe, since it's supposed to be b/w -1 and 1?)
    "NDVI": (-9821.0, 9996.0, 5157.625, 2466.6677),  # min, max
    # Precipitation in mm.
    # Negative values do not make sense, so min is set to 0.
    # 0., 99.9 percentile
    "pr": (0.0, 44.53038024902344, 1.7398051, 4.482833),
    # Specific humidity.
    # Negative values do not make sense, so min is set to 0.
    # The range of specific humidity is up to 100% so max is 1.
    "sph": (0.0, 1.0, 0.0071658953, 0.0042835088),
    # Wind direction in degrees clockwise from north.
    # Thus min set to 0 and max set to 360.
    "th": (0.0, 360.0, 190.32976, 72.59854),
    # Min/max temperature in Kelvin.
    # Min temp
    # -20 degree C, 99.9 percentile
    "tmmn": (253.15, 298.94891357421875, 281.08768, 8.982386),
    # Max temp
    # -20 degree C, 99.9 percentile
    "tmmx": (253.15, 315.09228515625, 295.17383, 9.815496),
    # Wind speed in m/s.
    # Negative values do not make sense, given there is a wind direction.
    # 0., 99.9 percentile
    "vs": (0.0, 10.024310074806237, 3.8500874, 1.4109988),
    # NFDRS fire danger index energy release component expressed in BTU's per
    # square foot.
    # Negative values do not make sense. Thus min set to zero.
    # 0., 99.9 percentile
    "erc": (0.0, 106.24891662597656, 37.326267, 20.846027),
    # Population density
    # min, 99.9 percentile
    "population": (0.0, 2534.06298828125, 25.531384, 154.72331),
    # 1 indicates fire, 0 no fire, -1 unlabeled data
    "PrevFireMask": (-1.0, 1.0, 0.0, 1.0),
    "FireMask": (-1.0, 1.0, 0.0, 1.0),
}