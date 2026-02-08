import numpy as np

TIMESTEPS = 60
SENSORS = 27
TOTAL_FEATURES = 1620

def build_time_series(sensor_values):
    """
    sensor_values: list of 27 sensor readings
    returns: np.array of shape (1, 1620)
    """

    sensor_values = np.array(sensor_values)

    if sensor_values.shape[0] != SENSORS:
        raise ValueError("Expected 27 sensor values")

    series = []

    for t in range(TIMESTEPS):
        decay = np.exp(-t / 10)     # temporal decay
        noise = np.random.normal(0, 0.01, SENSORS)
        timestep_values = sensor_values * decay + noise
        series.append(timestep_values)

    series = np.array(series).flatten()  # (1620,)
    return series.reshape(1, TOTAL_FEATURES)
