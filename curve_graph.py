import math
# import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Annotation
from matplotlib.legend import Legend
from matplotlib import rcParams
# from matplotlib import figure
import numpy as np
import Bezier
from pandas import read_csv
from pandas import date_range as p_date_range
from PIL import Image, ImageDraw
# import cv2.cv2 as cv
import get_parameter as para
import os
from tkinter.filedialog import askopenfilename
from tkinter import Tk, Button


class Draw:
    def __init__(self, fp):
        """
        # 从表格获取的数据会经过几个转换阶段
        # 原始数据: 未经任何处理
        # 补充数据: 对原始数据空缺时间进行补充后的数据
        # 平滑数据: 对补充数据用贝塞尔曲线连接后的点的数据
        # 曲线显示机制解释
        # 画出完整的曲线, 由line_clip_path剪切出要显示的范围
        # line_clip_path的左端以及右端位置从x_lim中选取
        # line_clip_path的左端位置称为帧刷新位置
        """
        self.fp = fp
        self.date_code_dict = para.date_code_dict
        self.y_m_d_dict = para.y_m_d_dict
        self.ini = para.ini
        self.config_path = para.config_path
        self.config = para.config
        self.rcParams_dict = para.rcParams_dict
        self.ico_path_dict = para.ico_path_dict
        self.ico_type = para.ico_type
        self.marker_style = para.marker_style
        self.ico_zoom = para.ico_zoom
        self.fig_size = para.fig_size
        self.axes_rect = para.axes_rect
        self.axes_frame_on = para.axes_frame_on
        self.point_div = para.point_div
        self.right_space = para.right_space
        self.xlabel_prefix = para.xlabel_prefix
        self.xlabel_suffix = para.xlabel_suffix
        self.xlabel_rotation = para.xlabel_rotation
        self.major_xlabel = para.major_xlabel
        self.major_xlabel_code = para.major_xlabel_code
        self.max_xsection_stage2 = para.max_xsection_stage2
        self.max_xsection_stage3 = para.max_xsection_stage3
        self.y_lim = para.y_lim
        self.y_label_format = para.y_label_format
        self.ylabel_prefix = para.ylabel_prefix
        self.ylabel_suffix = para.ylabel_suffix
        self.x_unit = para.x_unit
        self.x_unit_position = para.x_unit_position
        self.y_unit = para.y_unit
        self.y_unit_position = para.y_unit_position
        self.legend_on = para.legend_on
        self.legend_color = para.legend_color
        self.y_grid_on = para.y_grid_on
        self.x_grid_on = para.x_grid_on
        self.y_grid_alpha = para.y_grid_alpha
        self.x_grid_alpha = para.x_grid_alpha
        self.y_grid_linestyle = para.y_grid_linestyle
        self.x_grid_linestyle = para.x_grid_linestyle
        self.line_color_dict = para.line_color_dict
        self.auto_fill = para.auto_fill
        self.bezier_smoothness = para.bezier_smoothness
        self.bezier_auto_rotate = para.bezier_auto_rotate
        self.bezier_auto_scale = para.bezier_auto_scale
        self.bezier_scale_range = para.bezier_scale_range
        self.frame_div = para.frame_div
        self.video_fps = para.video_fps
        self.video_type = para.video_type
        self.synchro = para.synchro
        self.ffmpeg_path = para.ffmpeg_path
        self.video_output_path = para.video_output_path
        self.axes_setattr = para.axes_setattr
        self.video_finish = False

        print('图表生成中...')

        self._set_rc_params()

        # 从表格获取信息
        self.data_dict, self.date_list, self.statistic_dict = self.get_data(self.auto_fill, self.fp)
        self.continue_days = len(self.date_list)

        # self.fig: figure.Figure
        self.fig = plt.figure(figsize=self.fig_size)
        # self.ax: matplotlib.figure.Axes
        self.ax = self.fig.add_axes(self.axes_rect)
        self._set_axes_attr()

        self._calc_x_axis()

        # 画曲线及曲线相关的artist创建
        self.min_num, self.max_num = self._create_line_artists()

        # axes的artist创建
        self._create_else_artist()

        self._calc_frame()

        # 获取图片大小
        self._calc_video_size()

    def _set_rc_params(self):
        """
        设置rcParams
        :return:
        """
        for key, value in self.rcParams_dict.items():
            if value:
                try:
                    rcParams[key] = value
                except ValueError:
                    print('键值对{}={}错误'.format(key, value))

    def _set_axes_attr(self):
        """
        [setattr]
        :return:
        """
        for _key, _value in self.axes_setattr.items():
            attr = getattr(self.ax, _key, None)
            if attr:
                try:
                    attr(*_value['args'], **_value['kw'])
                except:
                    print('[setattr]: axes.{}(*args={}, **kw={})错误'.format(_key, _value['args'], _value['kw']))
            else:
                print('[setattr]: axes.{}对象不存在'.format(_key))

    def _calc_x_axis(self):
        """
        # 任何与x轴有关的方法都必须在此方法之后才能调用
        # 从表格获取的数据会经过几个转换阶段
        # 原始数据: 未经任何处理
        # 补充数据: 对原始数据空缺时间进行补充后的数据
        # 平滑数据: 对补充数据用贝塞尔曲线连接后的点的数据
        # 曲线显示机制解释
        # 画出完整的曲线, 由line_clip_path剪切出要显示的范围
        # line_clip_path的左端以及右端位置从x_lim中选取
        # line_clip_path的左端位置称为帧刷新位置
        """
        # 设置x轴的缩放系数和偏移系数, 目的是为了使x轴和y轴在数值尺度上相似
        self.x_scale = self.statistic_dict['最大值'] - self.statistic_dict['最小值']
        self.x_offset = self.statistic_dict['最小值']

        # 自动设置x轴刻度标签的时间间隔
        if self.major_xlabel_code == self.date_code_dict['A']:
            if self.continue_days > self.y_m_d_dict[self.statistic_dict['date_interval']] * 2 \
                    and self.statistic_dict['date_interval'] != 'Y':
                self.major_xlabel_code = self.date_code_dict[self.statistic_dict['date_interval']] + 1

        # 设置画面中最多出现的时间数max_all_xsection
        # 由major_xlabel_code和max_xsection_stage2确定
        self.xlabel_interval_code = self.major_xlabel_code - self.date_code_dict[self.statistic_dict['date_interval']]
        self.max_all_xsection = self.max_xsection_stage2
        if self.xlabel_interval_code == 1:
            for key, value in self.date_code_dict.items():
                if value == self.major_xlabel_code - 1:
                    self.max_all_xsection = self.max_xsection_stage2 * self.y_m_d_dict[key]
                    self.right_space *= self.y_m_d_dict[key]
        # 防止画面中最多出现的时间数超过表格中的总时间数
        if self.max_all_xsection > self.continue_days:
            self.max_all_xsection = self.continue_days

        self.old_x = np.arange(0, self.continue_days)  # x轴变换前的原始数据x位置对应关系
        self.old_x_calc = self.xaxis_trans(self.old_x)  # x轴变换后的原始数据x位置对应关系
        self.new_x = np.arange(0, self.continue_days - 1 + self.point_div, self.point_div)  # x轴变换前的平滑数据x位置对应关系
        self.new_x_calc = self.xaxis_trans(self.new_x)  # x轴变换后的平滑数据x位置对应关系
        self.right_new_x = [0, 1]  # 用于计算当前曲线末端图标在哪两个日期之间
        self.right_new_x_max = len(self.new_x) - 1

        self.x_lim = np.arange(0,
                               self.continue_days - 1 + self.frame_div + self.right_space,
                               self.frame_div)  # x轴变换前的帧刷新位置, 同时也是x轴范围的选取位置
        self.x_lim_calc = self.xaxis_trans(self.x_lim)  # x轴变换后的帧刷新位置, 同时也是x轴范围的选取位置
        if self.synchro:
            self.new_x = np.arange(0, self.continue_days - 1 + self.frame_div, self.frame_div)
            self.new_x_calc = self.xaxis_trans(self.new_x)

    def _create_line_artists(self):
        """
        必须在_calc_x_axis之后
        计算平滑数据中的最大值与最小值
        创建Line2D artist到data_dict, 并画曲线
        创建曲线末端图标(line marker)artist到data_dict
        :return:
        """
        one_data = self.data_dict.values().__iter__().__next__()
        max_num = one_data['数据'][0]
        min_num = max_num
        for key_, value_ in self.data_dict.items():
            xy = list(zip(self.old_x_calc, value_['数据']))

            bezier = Bezier.ConsecutiveBezier(xy,
                                              smoothness=self.bezier_smoothness,
                                              prop_range=self.bezier_scale_range,
                                              auto_ctrl_rotate=self.bezier_auto_rotate,
                                              auto_ctrl_prop=self.bezier_auto_scale)
            bezier_x, bezier_y = bezier.consecutive_bezier_x_y()

            _max = max(bezier_y)
            if _max > max_num:
                max_num = _max

            _min = min(bezier_y)
            if _min < min_num:
                min_num = _min

            new_y = np.array(Bezier.Coordinate.equal_distribute(bezier_x, bezier_y, self.new_x_calc))

            self.data_dict[key_]['数据'] = new_y
            line_color = self.line_color_dict.get(key_, None)
            self.data_dict[key_]['Line2D'], = self.ax.plot(bezier_x,
                                                           bezier_y,
                                                           color=line_color)
            if self.ico_type == 'image':
                self.data_dict[key_]['offset ico'] = self.create_offset_ico(self.ico_path_dict.get(str(key_), None),
                                                                            zoom=self.ico_zoom)
                self.data_dict[key_]['line marker'] = AnnotationBbox(self.data_dict[key_]['offset ico'],
                                                                     (1, 1),
                                                                     frameon=False,
                                                                     annotation_clip=False) \
                    if self.data_dict[key_]['offset ico'] else None
            elif self.ico_type == 'marker':
                self.data_dict[key_]['line marker'] = Line2D([0],
                                                             [0],
                                                             marker=self.marker_style,
                                                             markerfacecolor=self.data_dict[key_]['Line2D'].get_color(),
                                                             markeredgecolor=self.data_dict[key_]['Line2D'].get_color())
            else:
                self.data_dict[key_]['line marker'] = None
        return min_num, max_num

    def _create_else_artist(self):
        """
        设置其他的artist
        :return:
        """
        self.artists_dict = {'add': dict(), 'else': dict()}
        # 创建曲线剪切路径
        self.artists_dict['else']['line clip path'] = Rectangle((0, self.min_num - 5),
                                                                1,
                                                                self.max_num - self.min_num + 10,
                                                                transform=self.ax.transData)
        if self.x_unit:
            self.artists_dict['add']['x unit'] = Annotation(
                text=self.x_unit,
                xy=self.x_unit_position
                if self.x_unit_position[0] == [-1.0, -1.0]
                else (self.axes_rect[0] + self.axes_rect[2] - (1 / self.fig_size[0] * 0.2),
                      self.axes_rect[1] - (1 / self.fig_size[1] * 0.2)),
                xycoords=self.fig)

        if self.y_unit:
            self.artists_dict['add']['y unit'] = Annotation(
                text=self.y_unit,
                xy=self.y_unit_position
                if self.y_unit_position[0] == [-1.0, -1.0]
                else (self.axes_rect[0] - (1 / self.fig_size[0] * 0.2),
                      self.axes_rect[1] + self.axes_rect[3]),
                xycoords=self.fig)

        if self.legend_on:
            self.artists_dict['add']['legend'] = Legend(parent=self.ax,
                                                        handles=tuple(_value['Line2D']
                                                                      for _value in self.data_dict.values()),
                                                        labels=tuple(str(_key) for _key in self.data_dict.keys()),
                                                        labelcolor='linecolor' if self.legend_color else None,
                                                        loc='upper right')

    def _calc_frame(self):
        """
        计算各个阶段的frame
        :return:
        """
        #
        self.max_frames_real_time = int((self.max_all_xsection - 1) / self.frame_div) + 1  # 每幅图像最多包含的帧刷新位置数目
        self.right_space_frames = int(self.right_space / self.frame_div)  # 右侧留白包含的帧刷新位置数目
        self.all_frames_loc = len(self.x_lim) - self.right_space_frames  # 帧刷新位置的总数目

        # stage解释:
        # stage1 为由初始空白从左到右画至右端的过程
        # stage2 为x轴向左移动的过程
        # stage3 为画到最后一个点之后的x轴压缩过程
        # 计算各个stage的帧区间
        self.stage1_start_frame = 0
        self.stage1_continue_frame = min(self.max_frames_real_time, self.all_frames_loc)
        self.stage1_end_frame = self.stage1_continue_frame - 1

        self.stage2_start_frame = self.stage1_end_frame + 1
        self.stage2_continue_frame = self.all_frames_loc - self.max_frames_real_time
        self.stage2_end_frame = self.stage2_start_frame + self.stage2_continue_frame - 1

        self.stage3_start_frame = self.stage2_end_frame + 1
        self.stage3_continue_frame = self.all_frames_loc - self.max_frames_real_time
        self.stage3_end_frame = self.stage3_start_frame + self.stage3_continue_frame - 1

        self.all_frames = max(self.stage1_end_frame, self.stage3_end_frame) + 1
        self._all_frames_digit = 0
        self._all_frames_i = self.all_frames
        while self._all_frames_i >= 1:
            self._all_frames_digit += 1
            self._all_frames_i = self._all_frames_i / 10

    def _calc_video_size(self):
        """
        获取视频大小
        :return:
        """
        # 获取图片大小
        self.canvas = self.fig.canvas
        self.canvas.draw()
        self.video_size = np.shape(np.array(self.canvas.buffer_rgba()))[:2][::-1]

    def start(self):
        """
        开始
        :return:
        """
        self._animation()

    def _animation(self):
        """
        动态显示FuncAnimation
        :return:
        """
        # 动态显示
        ani = FuncAnimation(fig=self.fig,
                            func=self._ani_update,
                            init_func=self._ani_init,
                            interval=1000 / self.video_fps,
                            frames=np.arange(0, self.all_frames + 1),
                            repeat=False)

        if self.video_type == 'avi':
            # self.video_writer = VideoWriter(self.video_output_path,
            #                                 VideoWriter_fourcc('X', 'V', 'I', 'D'),
            #                                 self.video_fps,
            #                                 self.video_size)
            self.video_output_path = './files/statistic.avi'
        elif self.video_type == 'html':
            with open('./files/visualization.html', 'w') as f:
                f.write(ani.to_jshtml(fps=self.video_fps))
        elif self.video_type == 'mp4':
            rcParams['animation.ffmpeg_path'] = self.ffmpeg_path
            ani.save('./files/visualization.mp4')
        else:
            pass
        plt.show()

    def _ani_init(self):
        """
        FuncAnimation的初始化函数
        :return:
        """
        self.right_new_x = [0, 1]

        # 坐标轴右侧以及顶端的边框是否显示
        if not self.axes_frame_on:
            self.ax.spines['top'].set_color('none')
            self.ax.spines['right'].set_color('none')

        # 设置y轴范围
        if self.y_lim[0] == 0 and self.y_lim[1] == 0:
            pass
        else:
            self.ax.set_ylim(self.y_lim[0], self.y_lim[1])

        # 设置y轴刻度及刻度标签
        ylim = self.ax.get_ylim()
        y_ticks = self.ax.get_yticks()
        y_label_format_split = self.y_label_format.split('.')
        if len(y_label_format_split) == 0:
            pass
        elif len(y_label_format_split) == 2:
            format_int = int(y_label_format_split[0]) if y_label_format_split[0].isdecimal() else 0
            format_float = int(y_label_format_split[1]) if y_label_format_split[1].isdecimal() else -1
            format_str = '%.{}f'.format(format_float)
            y_labels = tuple(
                (format_str % (tick / 10 ** format_int) if format_str != '%.-1f' else tick / 10 ** format_int)
                for tick in y_ticks)
            plt.yticks(y_ticks, y_labels)

        if self.ylabel_suffix or self.ylabel_prefix:
            y_labels = tuple(''.join((self.ylabel_prefix, text.get_text(), self.ylabel_suffix))
                             for text in self.ax.get_yticklabels())
            plt.yticks(y_ticks, y_labels)
        self.ax.set_ylim(ylim[0], ylim[1])  # 由于设置y轴刻度及刻度标签后会自动重新设置y轴范围, 所以重新设置y轴范围

        # 设置x轴刻度及刻度标签, x轴范围
        x_lim0 = 0
        x_lim1 = self.max_all_xsection - 1 + self.right_space
        self._set_xticks(0, self.x_lim[-1], rotation=self.xlabel_rotation)
        self.ax.set_xlim(self.xaxis_trans(x_lim0), self.xaxis_trans(x_lim1))

        # 设置网格线grid
        if self.y_grid_on:
            plt.grid(linestyle=self.y_grid_linestyle, axis='y', alpha=self.y_grid_alpha)
        if self.x_grid_on:
            plt.grid(linestyle=self.x_grid_linestyle, axis='x', alpha=self.x_grid_alpha)

        # 添加所有的artist到axes
        for _value in self.data_dict.values():
            if _value.get('line marker', None):
                self.ax.add_artist(_value['line marker'])
        for artist in self.artists_dict['add'].values():
            if artist:
                self.ax.add_artist(artist)
                artist.set_clip_on(False)

    def _ani_update(self, frame):
        """
        FuncAnimation的更新函数
        :param frame:
        :return:
        """
        if frame <= self.stage1_end_frame:
            self._ani_stage1(frame)
        elif frame <= self.stage2_end_frame:
            self._ani_stage2(frame)
        elif frame <= self.stage3_end_frame:
            self._ani_stage3(frame)

        if self.video_type == 'png':
            # 保存为图片
            self.fig.savefig('./files/%s.png' % str(frame).zfill(self._all_frames_digit), transparent=True)
        if not self.video_finish:
            percentage = frame / self.all_frames * 100
            print('\r{:.1f}%'.format(percentage), end='')
            if percentage == 100:
                print('\n视频输出完成')
        if frame == self.all_frames:
            self.video_finish = True

    def _ani_stage1(self, frame):
        """
        阶段1
        :param frame:
        :return:
        """
        x0 = 0
        x1 = self.x_lim[frame]

        x0_calc = self.xaxis_trans(x0)
        x1_calc = self.xaxis_trans(x1)

        self._set_line_marker(x1, x1_calc)

        if self.synchro:
            for _value in self.data_dict.values():
                _value['Line2D'].set_data(self.new_x_calc[:frame + 1], _value['数据'][:frame + 1])
        else:
            self._set_line_clip(x0_calc, x1_calc)

    def _ani_stage2(self, frame):
        """
        阶段2
        :param frame:
        :return:
        """
        # x轴变换前曲线的左右端位置
        x0 = self.x_lim[frame - self.max_frames_real_time + 1]
        x1 = self.x_lim[frame]

        # x轴变换后曲线的左右端位置
        x0_calc = self.xaxis_trans(x0)
        x1_calc = self.xaxis_trans(x1)

        # 设置line marker
        self._set_line_marker(x1, x1_calc)

        # x轴变换前的x轴左右两端值
        x_lim0 = x0
        x_lim1 = x1 + self.right_space

        # 设置x轴的刻度及刻度标签
        self._set_xticks(x_lim0, x_lim1, rotation=self.xlabel_rotation)
        # 设置x轴范围
        self.ax.set_xlim(self.xaxis_trans(x_lim0), self.xaxis_trans(x_lim1))

        if self.synchro:
            # 曲线绘制与帧刷新同步, 设置曲线data
            for _value in self.data_dict.values():
                _value['Line2D'].set_data(self.new_x_calc[:frame + 1],
                                          _value['数据'][:frame + 1])
        else:
            # 不同步, 设置剪切范围
            self._set_line_clip(x0_calc, x1_calc)

    def _ani_stage3(self, frame):
        """
        阶段3
        :param frame:
        :return:
        """
        x0 = self.x_lim[self.all_frames_loc - self.max_frames_real_time + self.all_frames_loc - 1 - frame]
        x1 = self.new_x[-1]

        x0_calc = self.xaxis_trans(x0)
        x1_calc = self.xaxis_trans(x1)

        x_lim0 = x0
        x_lim1 = self.x_lim[-1]
        self._set_xticks(x_lim0, x_lim1, rotation=self.xlabel_rotation, max_xsection=self.max_xsection_stage3)
        self.ax.set_xlim(self.xaxis_trans(x_lim0), self.xaxis_trans(x_lim1))

        if frame == self.stage3_start_frame:
            for _value in self.data_dict.values():
                if _value['line marker']:
                    _value['line marker'].xyann = (self.new_x_calc[-1], _value['数据'][-1])

        # 由于同步设置时未设置曲线剪切范围, matplotlib会自动设置曲线的显示范围, 因此此处不必设置曲线的data
        if not self.synchro:
            self._set_line_clip(x0_calc, x1_calc)

    def _set_xticks(self, xlim0, xlim1, rotation=0, max_xsection=None):
        """
        设置x轴刻度及刻度标签
        :param xlim0:           x轴范围的左端值(变换前)
        :param xlim1:           x轴范围的右端值(变换前)
        :param rotation:        刻度标签旋转角度
        :param max_xsection:    最多显示刻度数目
        :return:
        """
        x_ticks_l = math.ceil(xlim0)
        x_ticks_r = math.floor(xlim1) + 1
        while x_ticks_r > self.continue_days:
            x_ticks_r -= 1
        x_ticks = np.arange(x_ticks_l, x_ticks_r, 1)
        x_labels = list(map(lambda ii: self.date_list[ii], x_ticks))

        if self.xlabel_interval_code == 1:
            new_x_ticks = []
            new_x_labels = []
            for index, label in enumerate(x_labels):
                if label[-2:] == '01':
                    new_x_ticks.append(x_ticks[index])
                    new_x_labels.append(''.join((self.xlabel_prefix, label[:-3], self.xlabel_suffix)))

        else:
            new_x_ticks = x_ticks
            new_x_labels = x_labels
        if max_xsection is not None:
            div_num = math.ceil(len(new_x_ticks) / (max_xsection + 1))
            index = np.arange(len(new_x_ticks) - 1, -1, -div_num)
            new_x_ticks = list(map(lambda ii: new_x_ticks[ii], index))
            new_x_labels = list(map(lambda ii: new_x_labels[ii], index))

        plt.xticks(self.xaxis_trans(new_x_ticks), new_x_labels, rotation=rotation)

    def _set_line_clip(self, x0_calc, x1_calc):
        """
        设置曲线剪切路径
        :param x0_calc:     剪切路径左端位置(x轴变换后)
        :param x1_calc:     剪切路径右端位置(x轴变换后)
        :return:
        """
        for _value in self.data_dict.values():
            self.artists_dict['else']['line clip path'].set_x(x0_calc)
            self.artists_dict['else']['line clip path'].set_width(x1_calc - x0_calc)
            _value['Line2D'].set_clip_path(self.artists_dict['else']['line clip path'])

    def _set_line_marker(self, x1, x1_calc):
        """
        设置line marker的位置
        :param x1:          x轴位置(x轴变换前)
        :param x1_calc:     x轴位置(x轴变换后)
        :return:
        """
        # x1_calc = xaxis_trans(x1)
        while (not ((x1 >= self.new_x[self.right_new_x[0]]) and (x1 <= self.new_x[self.right_new_x[1]]))) and \
                (self.right_new_x[1] < self.right_new_x_max):
            self.right_new_x[0] = self.right_new_x[1]
            self.right_new_x[1] += 1
        for _value in self.data_dict.values():
            left_y = _value['数据'][self.right_new_x[0]]
            right_y = _value['数据'][self.right_new_x[1]]
            left_x = self.new_x[self.right_new_x[0]]
            right_x = self.new_x[self.right_new_x[1]]
            y1 = (x1 - left_x) / (right_x - left_x) * (right_y - left_y) + left_y
            if _value.get('line marker', None):
                if isinstance(_value['line marker'], AnnotationBbox):
                    _value['line marker'].xyann = (x1_calc, y1)
                else:
                    _value['line marker'].set_xdata(x1_calc)
                    _value['line marker'].set_ydata(y1)

    @staticmethod
    def annotation_box(img_box, position):
        """
        创建一个存放img_box的AnnotationBbox
        :param img_box:     存放image的OffsetBox
        :param position:    位置
        :return:
        """
        if all((img_box, position)):
            ab = AnnotationBbox(img_box, position, frameon=False)
            return ab
        return None

    @staticmethod
    def create_offset_ico(ico_path, zoom=1):
        """
        缩放裁剪图标, 并将图标存放在一个OffsetBox内
        :param zoom:        float: 图标缩放
        :param ico_path:    图标路径
        :return:            存放img的OffsetBox
        """
        if not ico_path or not os.path.exists(ico_path):
            return None

        ico_size = 100

        img: Image.Image
        img = Image.open(ico_path)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        img_size = img.size
        if img_size[0] != img_size[1]:
            img_shortest_len = img_size[0] if img_size[0] <= img_size[1] else img_size[1]
            crop_half_len = int(img_shortest_len / 2)
            img_center = (int(img_size[0] / 2), int(img_size[1] / 2))

            img = img.crop((img_center[0] - crop_half_len,
                            img_center[1] - crop_half_len,
                            img_center[0] + crop_half_len,
                            img_center[1] + crop_half_len))

        if img.size[0] != ico_size:
            ico = img.resize((ico_size, ico_size))
        else:
            ico = img

        mask = Image.new('L', [ico_size, ico_size], 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse((0, 0, ico_size, ico_size), fill=255)
        ico.putalpha(mask)

        return OffsetImage(ico, zoom=0.2 * zoom)

    def xaxis_trans(self, x):
        """
        x轴坐标变换
        :param x:
        :return:
        """
        if isinstance(x, (int, float)):
            pass
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return x * self.x_scale + self.x_offset

    @staticmethod
    def get_data(auto_fill, path):
        """
        从表格获取数据
        对空缺时间与数据进行填充
        计算一些统计数据
        :param auto_fill:   bool: 是否自动填充丢失的数据
        :param path:        excel文件路径
        :return:            dict: 填充后的数据{name: date, ...}
                            list: 填充后的时间序列
                            dict: 储存一些统计数据的字典
        """
        # original_df: pandas.DataFramey=['日期'], ascending=True)
        original_df = read_csv(path)
        original_df = original_df.sort_values(by=['日期'])
        # date_type = original_df['日期'].dtype
        # if pandas.api.types.is_datetime64_any_dtype(date_type):
        #     original_df['日期'] = original_df['日期'].dt.strftime(date_format='%Y-%m-%d')
        _statistic_dict = dict()
        _statistic_dict['平均值'] = original_df['数据'].mean()
        _statistic_dict['最大值'] = original_df['数据'].max()
        _statistic_dict['最小值'] = original_df['数据'].min()

        # 通过名字分类
        group_by_name_df = original_df.groupby(by=['名字'])
        name_list = list(group_by_name_df.groups.keys().__iter__())

        first_date = group_by_name_df.get_group(name_list[0])['日期'].values
        start_date = first_date[0]
        end_date = first_date[-1]

        len_date = len(start_date)
        if len_date == 7:
            date_interval = 'MS'
        elif len_date == 4:
            date_interval = 'YS'
        else:
            date_interval = 'D'
        _statistic_dict['date_interval'] = date_interval  # 时间间隔

        # 获取表格中的最早日期与最晚日期
        for name in name_list:
            _date = group_by_name_df.get_group(name)['日期'].values
            _start_date = _date[0]
            _end_date = _date[-1]
            if _start_date < start_date:
                start_date = _start_date
            if _end_date > end_date:
                end_date = _end_date

        # 日期填充
        new_date = [(str(d) if date_interval == 'D' else str(d)[:len_date])
                    for d in p_date_range(start=start_date, end=end_date, freq=date_interval).to_period()]

        # 数据填充
        days = len(new_date)
        group_by_name_dict = dict()
        for name in name_list:
            type_df = group_by_name_df.get_group(name)
            original_date = type_df['日期'].values
            original_data = type_df['数据'].values

            if not auto_fill:
                group_by_name_dict[name] = {'数据': original_data}
                continue

            # 日期总数不对则有空缺
            if days != len(type_df['日期'].values):
                ref_points = []
                original_date_index = 0

                # 由贝塞尔曲线进行数据填充
                # 创建参考点
                for i in range(days):
                    if i == 0 and new_date[0] != original_date[0]:
                        ref_points.append((0, original_data[0]))
                    elif original_date_index >= (len(original_date) - 1):
                        ref_points.append((days - 1, ref_points[-1][1]))
                        break
                    elif new_date[i] == original_date[original_date_index]:
                        ref_points.append((i, original_data[original_date_index]))
                        original_date_index += 1
                # 计算贝塞尔曲线
                bezier_points = Bezier.ConsecutiveBezier(ref_points).consecutive_bezier_x_y()
                new_data = Bezier.Coordinate.equal_distribute(bezier_points[0], bezier_points[1], np.arange(0, days))
                group_by_name_dict[name] = {'数据': new_data}
            else:
                group_by_name_dict[name] = {'数据': type_df['数据'].values}
        return group_by_name_dict, new_date, _statistic_dict


class Win:
    def __init__(self, win_width, win_height):
        self.excel_path = ''
        self.win = Tk()
        self.win_width = win_width
        self.win_height = win_height

    def start(self):
        screen_width = self.win.winfo_screenwidth()
        screen_height = self.win.winfo_screenheight()
        location_size = '%dx%d+%d+%d' % (self.win_width, self.win_height, (screen_width - self.win_width) / 2,
                                         (screen_height - self.win_height) / 2)
        self.win.geometry(location_size)
        self.win.resizable(0, 0)
        select_file_btn = Button(self.win, text='选择文件', command=self.select_file_cb)
        # select_file_btn.pack(padx=15, pady=10)
        btn_width = 80
        btn_height = 30
        select_file_btn.place(x=(self.win_width - btn_width) / 2,
                              y=(self.win_height - btn_height) / 2,
                              width=btn_width,
                              height=btn_height)
        self.win.mainloop()

    def select_file_cb(self):
        self.excel_path = askopenfilename()
        if self.excel_path:
            self.win.destroy()


if __name__ == '__main__':
    win = Win(200, 120)
    win.start()
    excel_path = win.excel_path
    if excel_path:
        chart = Draw(excel_path)
        chart.start()
