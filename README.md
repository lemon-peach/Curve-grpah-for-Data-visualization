# Curve-grpah-for-Data-visualization
## 介绍
此项目为曲线图形式的动态数据可视化工具，能够很好的显示数据随时间的变化趋势，支持多种自定义设置
## 使用
运行curve_graph.py，选择存储数据的csv文件，程序运行完毕后结果会保存在files文件夹下

csv文件的列名必须为“名字”、“数据”、“日期”，顺序可以颠倒，日期必须为“YYYY-mm-dd”形式，编码方式为utf-8。具体形式参考example.csv文件
### 自定义设置
通过修改config文件夹下的config.ini文件进行自定义设置

## 环境
python3
### 需要的库
matplotlib, pandas, numpy

## 文件说明
Bezire.py           计算多个点之间的贝塞尔曲线以及其他一些坐标变换的相关函数
get_parameter.py    读取config.ini文件中的参数
curve_graph.py      画曲线图
/config/config.ini  通过修改参数实现曲线图的自定义设置

## 联系方式
电子邮件: lemon8peach@gmail.com
