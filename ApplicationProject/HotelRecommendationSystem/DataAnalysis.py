import pandas as pd
import missingno as msno

from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')

if __name__ == '__main__':
    df = pd.read_csv('data/hotel_bookings.csv')
    # print(df.describe())
    # print(df.info())
    # print(df.head())
    null_data = pd.DataFrame({
        'Null Values': df.isna().sum(),
        'Percentage Null Values': (df.isna().sum()) / (df.shape[0]) * (100)
    })
    # print(null_data)
    df.fillna(0, inplace=True)
    # print(df.isna().sum())

    # msno.bar(df)
    # plt.show()

    country_wise_guests = df[df['is_canceled'] == 0]['country'].value_counts().reset_index()
    country_wise_guests.columns = ['country', 'No of guests']
    # print(country_wise_guests)
