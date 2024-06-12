import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from database import Database
from preprocessing import charge_format_timeSeries, denoising_time_series, filter_fft


def regression_lagValue(data: pd.DataFrame, as_plot: bool = True):
    df_lag = data.copy()
    df_lag["lag_1"] = df_lag["harga"].shift(1)
    df_lag.dropna(inplace=True)  # Remove rows with NaN values created by the shift
    if as_plot:
        plt.figure(figsize=(12, 6))
        sns.regplot(x="lag_1", y="harga", data=df_lag)
        plt.title("Regression plot with Lag Values")
        plt.xlabel("Lag 1 - harga")
        plt.ylabel("harga")
        plt.show()
    else:
        X = df_lag["lag_1"]
        y = df_lag["harga"]
        model = sm.OLS(y, X).fit()
        summary = model.summary()
        return summary


def main():
    db = Database("db_price_commodity.db")
    db.connect()
    query = """select Tanggal,round(avg(`Harga Mean`),4) as harga from berasmedium
    group by Tanggal;
    """
    df = db.execute_query(query, True)
    print(df)
    df_change = charge_format_timeSeries("Tanggal", y="harga", data=df, format="D")
    df_filtered = filter_fft(df_change, column="harga", threshold=1e8)
    # print(regression_lagValue(df_change,as_plot=True))
    denoising_time_series(df_filtered, "Tanggal", "harga", rolling_val=25, as_plot=True)
    # print(df_change)


if __name__ == "__main__":
    main()
