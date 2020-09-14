import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

fig, ax = plt.subplots()

months = mdates.MonthLocator()

dateFmt = mdates.DateFormatter("%m/%d/%y")

ax.xaxis.set_major_formatter(dateFmt)
ax.xaxis.set_minor_locator(months)
ax.tick_params(axis="both", direction="out", labelsize=10)

date1 = datetime.date(2005, 8, 8)
date2 = datetime.date(2015, 6, 6)
delta = datetime.timedelta(days=5)
dates = mdates.drange(date1, date2, delta)

y = np.random.normal(100, 15, len(dates))

ax.plot_date(dates, y, "#FF8800", alpha=0.7)

fig.autofmt_xdate()

plt.show()
