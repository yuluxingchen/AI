import numpy as np
import pandas as pd


class DataPreProcessing:
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(self.path)

    def pre_processing(self):
        df = self.df
        # 将空填充为0
        df.fillna(0, inplace=True)
        # 删除多余列
        useless_col = ['days_in_waiting_list',
                       'arrival_date_year',
                       'arrival_date_month',
                       'assigned_room_type',
                       'booking_changes',
                       'reservation_status',
                       'country']
        df.drop(useless_col, axis=1, inplace=True)
        # 将每列类型为字符串的索引筛选出来
        cat_cols = [col for col in df.columns if df[col].dtype == 'O']
        cat_df = df[cat_cols]
        # 将日期拆分为年月日
        date_df = pd.to_datetime(cat_df['reservation_status_date'])
        cat_df = pd.concat([cat_df,
                            date_df.dt.year,
                            date_df.dt.month,
                            date_df.dt.day],
                           axis=1)
        cat_df.columns = cat_cols + ['year', 'month', 'day']
        cat_df.drop(['reservation_status_date'], axis=1, inplace=True)
        cat_df['hotel'] = cat_df['hotel'].map({'Resort Hotel': 0, 'City Hotel': 1})
        cat_df['meal'] = cat_df['meal'].map({'BB': 0, 'FB': 1, 'HB': 2, 'SC': 3, 'Undefined': 4})
        cat_df['market_segment'] = cat_df['market_segment'].map(
            {'Direct': 0, 'Corporate': 1, 'Online TA': 2, 'Offline TA/TO': 3,
             'Complementary': 4, 'Groups': 5, 'Undefined': 6, 'Aviation': 7})
        cat_df['distribution_channel'] = cat_df['distribution_channel'].map(
            {'Direct': 0, 'Corporate': 1, 'TA/TO': 2, 'Undefined': 3,
             'GDS': 4})

        cat_df['reserved_room_type'] = cat_df['reserved_room_type'].map(
            {'A': 0, 'D': 1, 'E': 2, 'F': 3, 'G': 4, 'B': 5, 'C': 6,
             'H': 7, 'P': 8, 'L': 9})

        cat_df['deposit_type'] = cat_df['deposit_type'].map({'No Deposit': 0, 'Refundable': 1, 'Non Refund': 3})
        cat_df['customer_type'] = cat_df['customer_type'].map(
            {'Transient': 0, 'Contract': 1, 'Transient-Party': 2, 'Group': 3})
        cat_df['year'] = cat_df['year'].map({2015: 0, 2014: 1, 2016: 2, 2017: 3})

        cat_cols = [col for col in cat_df.columns if col in df.columns]
        num_df = df.drop(columns=cat_cols, axis=1)
        num_df.drop('is_canceled', axis=1, inplace=True)
        num_df.drop(['reservation_status_date'], axis=1, inplace=True)
        num_df['lead_time'] = np.log(np.where(num_df['lead_time'] + 1 > 0, num_df['lead_time'] + 1, np.nan))
        num_df['arrival_date_week_number'] = np.log(
            np.where(num_df['arrival_date_week_number'] + 1 > 0, num_df['arrival_date_week_number'] + 1, np.nan))
        num_df['arrival_date_day_of_month'] = np.log(
            np.where(num_df['arrival_date_day_of_month'] + 1 > 0, num_df['arrival_date_day_of_month'] + 1, np.nan))
        num_df['agent'] = np.log(np.where(num_df['agent'] + 1 > 0, num_df['agent'] + 1, np.nan))
        num_df['company'] = np.log(np.where(num_df['company'] + 1 > 0, num_df['company'] + 1, np.nan))
        num_df['adr'] = np.log(np.where(num_df['adr'] + 1 > 0, num_df['adr'] + 1, np.nan))
        num_df['adr'] = num_df['adr'].fillna(value=num_df['adr'].mean())
        result = [cat_df, num_df]
        return result, df['is_canceled']


if __name__ == '__main__':
    processing = DataPreProcessing('data/hotel_bookings.csv')
    dataset, label = processing.pre_processing()
    print(dataset)
    print(label)
    X = pd.concat(dataset, axis=1)
    print(X.dtypes)
    if X.isna().any().any():
        print("数据中包含无效值")
        pd.set_option('display.max_columns', None)
        nan_rows = X[X.isna().any(axis=1)]
        print("包含 NaN 值的行：")
        print(nan_rows)
        raise ValueError
    if X.map(np.isinf).any().any():
        print("数据中包含无穷大值")
        raise ValueError
