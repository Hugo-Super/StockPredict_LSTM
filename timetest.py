import datetime
import akshare as ak
# start_dt = '20100101'
# time_temp = datetime.datetime.now() - datetime.timedelta(days=1)
# end_dt = time_temp.strftime('%Y%m%d')
# print(end_dt)

stock_lhb_detail_daily_sina_df = ak.stock_lhb_detail_daily_sina(date="20240430")
print(stock_lhb_detail_daily_sina_df)