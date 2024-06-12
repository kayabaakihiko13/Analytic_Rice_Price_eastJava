import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq, irfft


def change_format_series(
    data: pd.DataFrame, date_column: str, value_column: str, format: str
) -> pd.DataFrame:
    if data is None:
        raise ValueError("No data available. Please fetch data first.")

    df_formatted = data.copy()

    # Ensure the date column is in datetime format
    df_formatted[date_column] = pd.to_datetime(df_formatted[date_column])

    # Set the date column as the index
    df_formatted.set_index(date_column, inplace=True)

    # Resample the DataFrame according to the specified format
    df_resampled = df_formatted.resample(format).mean()

    return df_resampled


def charge_format_timeSeries(
    x: str, y: str, data: pd.DataFrame, format: str
) -> pd.DataFrame:
    df_charge_formatted = data.copy()
    df_charge_formatted[x] = pd.to_datetime(df_charge_formatted[x])
    df_charge_formatted.set_index(x, inplace=True)
    return df_charge_formatted.resample(format).median()


def filter_fft(df: pd.DataFrame, column: str, threshold: float = 1e8) -> pd.DataFrame:
    df_filter = df.copy()

    # Perform FFT on the specified column
    fourier = rfft(df_filter[column])

    # Compute the corresponding frequencies
    n = df_filter[column].size
    timestep = 20e-3 / n
    frequencies = rfftfreq(n, d=timestep)

    # Filter out frequencies higher than the threshold
    fourier[frequencies > threshold] = 0

    # Perform inverse FFT to get the filtered time series
    filtered_signal = irfft(fourier)

    # Add the filtered signal to the DataFrame
    df_filter[column + "_filtered"] = filtered_signal

    return df_filter


def denoising_time_series(
    df: pd.DataFrame,
    DateColumn: str,
    X_col: str,
    rolling_val: int,
    as_plot: bool = True,
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Tipe data harus menggunakan DataFrame")

    if df.index.name != DateColumn:
        df = df.reset_index()

    rolling_df = df.copy()
    rolling_df["smoothed_" + X_col] = rolling_df[X_col].rolling(rolling_val).mean()

    if as_plot:
        plt.figure(figsize=(12, 6))
        plt.plot(df[DateColumn], df[X_col], label="Original", alpha=0.5)
        plt.plot(
            rolling_df[DateColumn],
            rolling_df["smoothed_" + X_col],
            label="Smoothed",
            color="red",
        )
        plt.title(f"Time Series Smoothing with Rolling Window of {rolling_val}")
        plt.xlabel("Tanggal")
        plt.ylabel("Harga")
        plt.legend()
        plt.show()

    return rolling_df
