import csv
from datetime import datetime
from matplotlib import pyplot as plt
 
filename = 'plant_traits_CM001.csv'  # 里面是模拟的2017年8月最高与最低温度值
with open(filename) as f:
    reader = csv.reader(f)
    header_row = next(reader)
    '''模块csv包含函数next()，调用它并将阅读器对象传递给它时，它将返回文件中的下一行。
    在前面的代码中，我们只调用了next()一次，因此得到的是文件的第一行，其中包含文件头。
    我们将返回的数据存储在header_ row中。
    '''
    print(header_row) #打印这一行，发现就是表头，因此NEXT的含义类似行指针走一行
    dates, inflorescence_widths, inflorescence_heights, stem_heights, plant_heights = [], [], [], [],[] #创建空列表
    for row in reader:
        current_date = datetime.strptime(row[1], "%Y-%m-%d") #datatime按日期格式转换
        inflorescence_width = float(row[2])   #第二列
        inflorescence_height = float(row[3])
        stem_height = float(row[4])
        plant_height = float(row[5])
        dates.append(current_date)
        inflorescence_widths.append(inflorescence_width)
        inflorescence_heights.append(inflorescence_height)
        stem_heights.append(stem_height)
        plant_heights.append(plant_height)


fig = plt.figure(dpi=128, figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(dates, inflorescence_heights, c='red', alpha=0.5)
title = "inflorescence_widths"
plt.title(title, fontsize=20)
plt.xlabel('', fontsize=16)
fig.autofmt_xdate()
plt.ylabel("Dimension (CM)", fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.legend()

plt.subplot(2,1, 2)
plt.plot(dates, inflorescence_widths, c='orange', alpha=0.5)
title = "inflorescence_height"
plt.title(title, fontsize=20)
plt.xlabel('', fontsize=16)
fig.autofmt_xdate()
plt.ylabel("Dimension (CM)", fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.show()
#plt.plot(dates, lows, c='orange', alpha=0.5)
#plt.fill_between(dates, highs, lows, facecolor='blue', alpha=0.1) #在两条折线着色
 
# Format plot.
