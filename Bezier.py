import numpy as np
import math


# class Bezier(object):
#     def __init__(self, points: list, smoothness: int):
#         self.points = points
#         self.smoothness = smoothness
#         self.t_all_num = self.smoothness + 2
#         self.space = 1 / (self.smoothness + 1)
#         self.t_tuple = tuple(i_f * self.space for i_f in range(self.t_all_num))
#         self.line_tuple = Coordinate.get_lines_by_points(self.points)
#
#     def bezier(self):
#         base_points = self.points
#         base_lines = Coordinate.get_lines_by_points(base_points)
#         t_points = list(map(lambda line: Coordinate.get_points_on_line(line, self.t_tuple), base_lines))
#         t_points = self.zip_list(t_points)
#         while True:
#             if len(t_points[0]) == 1:
#                 points = []
#                 for point in t_points:
#                     points.append(point[0])
#                 return points
#             t_lines = list(map(Coordinate.get_lines_by_points, t_points))
#             t_points = list(list((Coordinate.get_points_on_line(line, self.t_tuple[t])) for line in lines)
#                             for t, lines in enumerate(t_lines))
#
#     @staticmethod
#     def get_lines_by_points(points):
#         lines = []
#         for i_f in range(len(points) - 1):
#             lines.append([points[i_f], points[i_f + 1]])
#         return lines
#
#     def first_order(self, p1, p2):
#         pass
#
#     @staticmethod
#     def get_points_on_line(line, prop):
#         line = list(line)
#         p1_np = np.array(line[0])
#         p2_np = np.array(line[1])
#         if isinstance(prop, (int, float)):
#             p = p1_np + (p2_np - p1_np) * prop
#         else:
#             p = [p1_np + (p2_np - p1_np) * t for t in prop]
#         return p
#
#     @staticmethod
#     def zip_list(list_):
#         result = []
#         result_len = len(list_[0])
#         for j in range(result_len):
#             d = []
#             for i_f in list_:
#                 d.append(i_f[j])
#             result.append(d)
#         return result


class ConsecutiveBezier(object):

    def __init__(self, points: list,
                 smoothness=10,
                 ctrl_prop=1.0,
                 auto_ctrl_prop=True,
                 prop_range=None,
                 auto_ctrl_rotate=True,
                 ctrl_rotate=1):
        self.points = points
        self.smoothness = smoothness
        self.ctrl_prop = ctrl_prop
        self.auto_ctrl_prop = auto_ctrl_prop
        self.ctrl_rotate = ctrl_rotate
        if prop_range is None:
            self.prop_range = [0, 1]
        else:
            self.prop_range = prop_range
        self.auto_ctrl_rotate = auto_ctrl_rotate
        self.ctrl_points = []

    @staticmethod
    def bezier(points, smoothness):
        """
        由控制点points计算贝塞尔曲线, 计算方法为几何法
        :param points:          控制点
        :param smoothness:      贝塞尔曲线的平滑度, 即每两个控制点之间生成的点的数量
        :return:                贝塞尔曲线上的点的坐标
        """
        t_all_num = smoothness + 2  #
        space = 1 / (smoothness + 1)
        t_tuple = tuple(i_f * space for i_f in range(t_all_num))  # 比例的列表
        base_lines = Coordinate.get_lines_by_points(points)  # 控制点顺次连接生成的线段
        t_points = list(map(lambda line: Coordinate.get_points_on_line(line, t_tuple), base_lines))  # 获取线段上的比例点
        t_points = Coordinate.xy_to_x_y(t_points)  # 将比例点以比例作为第一索引
        while True:
            if len(t_points[0]) == 1:
                points = []
                for point in t_points:
                    points.append(point[0])
                return points
            t_lines = list(map(Coordinate.get_lines_by_points, t_points))  # 下一级线段
            t_points = list(list((Coordinate.get_points_on_line(line, t_tuple[t])) for line in lines)
                            for t, lines in enumerate(t_lines))  # 下一级点

    def ctrl(self, point_1, point0, point1, point_2=None, point2=None):
        """
        比例法求点point0的两个控制点坐标
        :param point_1:
        :param point0:
        :param point1:
        :param point_2:
        :param point2:
        :return:
        """
        rotate_scale_flag = False
        if point_2 and point2:
            rotate_scale_flag = True
        min_points = self.midpoint([point_1, point0, point1])
        min_point0 = min_points[0]
        min_point1 = min_points[1]
        line_length = self.points_distance([point_1, point0, point1])

        # 求比例点的坐标prop_point
        if line_length[1] == 0:
            prop_point = min_point1
        else:
            prop = line_length[0] / line_length[1]
            prop_point = Coordinate.get_points_on_line([min_point0, min_point1], prop / (prop + 1))

        ctrl_line0 = np.array(min_point0) - np.array(prop_point)
        ctrl_line1 = np.array(min_point1) - np.array(prop_point)

        if rotate_scale_flag:
            slopes_deg = Coordinate.slope([point_2, point_1, point0])
            relative_rotate = abs(slopes_deg[0] - slopes_deg[1]) / (2 * math.pi)
            ctrl_prop_00 = relative_rotate

            slopes_deg = Coordinate.slope([point0, point1, point2])
            relative_rotate = abs(slopes_deg[0] - slopes_deg[1]) / (2 * math.pi)
            ctrl_prop_10 = relative_rotate

            if self.auto_ctrl_rotate:
                ctrl_line0_deg = Coordinate.point_slope_angle(ctrl_line0)
                ctrl_line1_deg = Coordinate.point_slope_angle(ctrl_line1)

                if ctrl_prop_00 == 0:
                    if ctrl_prop_10 == 0:
                        ctrl_prop_0_1 = 0
                    else:
                        ctrl_prop_0_1 = -1
                elif ctrl_prop_10 == 0:
                    ctrl_prop_0_1 = 1
                else:
                    if ctrl_prop_00 < ctrl_prop_10:
                        ctrl_prop_0_1 = math.atan(ctrl_prop_10 / ctrl_prop_00) * -1
                        ctrl_prop_0_1 = (ctrl_prop_0_1 + math.pi / 4) / (math.pi / 4)
                    else:
                        ctrl_prop_0_1 = math.atan(ctrl_prop_00 / ctrl_prop_10)
                        ctrl_prop_0_1 = (ctrl_prop_0_1 - math.pi / 4) / (math.pi / 4)

                if ctrl_prop_0_1 <= 0:
                    angle = Coordinate.slope([point0, point_1]) - ctrl_line0_deg
                    if angle > math.pi:
                        angle = angle - math.pi * 2
                    elif angle < math.pi * -1:
                        angle = angle + math.pi * 2
                    rotate_angle = angle * ctrl_prop_0_1 * -1

                else:
                    angle = Coordinate.slope([point0, point1]) - ctrl_line1_deg
                    if angle > math.pi:
                        angle = angle - math.pi * 2
                    elif angle < math.pi * -1:
                        angle = angle + math.pi * 2
                    rotate_angle = angle * ctrl_prop_0_1 * self.ctrl_rotate

                left_line = (point_1[0] - point0[0], point_1[1] - point0[1])
                right_line = (point1[0] - point0[0], point1[1] - point0[1])

                ctrl_line0 = np.array(self._ctrl_rotate(ctrl_line0, rotate_angle, left_line))
                ctrl_line1 = np.array(self._ctrl_rotate(ctrl_line1, rotate_angle, right_line))

            if self.auto_ctrl_prop:
                ctrl_prop0 = self.map_range(ctrl_prop_00, (0, 1), self.prop_range)
                ctrl_prop1 = self.map_range(ctrl_prop_10, (0, 1), self.prop_range)
                ctrl_line0 = ctrl_line0 * ctrl_prop0
                ctrl_line1 = ctrl_line1 * ctrl_prop1

        ctrl_line0 = ctrl_line0 * self.ctrl_prop
        ctrl_line1 = ctrl_line1 * self.ctrl_prop
        ctrl0 = np.array(point0) + np.array(ctrl_line0)
        ctrl1 = np.array(point0) + np.array(ctrl_line1)
        return ctrl0, ctrl1

    def consecutive_bezier_xy(self):
        """
        计算贝塞尔曲线
        :return:
        """
        points_extend = [self.points[0]]
        points_extend.extend(self.points)
        points_extend.append(self.points[-1])
        point_num = len(self.points)
        bezier_ctrl_point = []
        result = []

        for point_index in range(1, point_num + 1):
            if (self.auto_ctrl_prop or self.auto_ctrl_rotate) and point_index != 1 and point_index != point_num:
                ctrl0, ctrl1 = self.ctrl(points_extend[point_index - 1], points_extend[point_index],
                                         points_extend[point_index + 1],
                                         points_extend[point_index - 2], points_extend[point_index + 2])
            else:
                ctrl0, ctrl1 = self.ctrl(points_extend[point_index - 1], points_extend[point_index],
                                         points_extend[point_index + 1])
            bezier_ctrl_point.extend([ctrl0, points_extend[point_index], ctrl1])
        bezier_ctrl_point = bezier_ctrl_point[1:-1]

        for line_index in range(point_num - 1):
            bezier_4_points = bezier_ctrl_point[(line_index * 3):(line_index * 3 + 4)]

            bezier = self.bezier(bezier_4_points, self.smoothness)
            if line_index == point_num - 2:
                result.extend(bezier)
                self.ctrl_points.extend(bezier_4_points)
            else:
                self.ctrl_points.extend(bezier_4_points[:-1])
                result.extend(bezier[:-1])
        return result

    def consecutive_bezier_x_y(self):
        """
        [[x1, y1], [x2, y2]...] --> [[x1, x2, ...], [y1, y2, ...]]
        :return:
        """
        return Coordinate.xy_to_x_y(self.consecutive_bezier_xy())

    @staticmethod
    def _ctrl_rotate(ctrl_line, angle, sideline):
        ctrl_line_rotate = Coordinate.rotate(ctrl_line, angle)

        if sideline[0] == 0:
            if ctrl_line[0] == 0:
                ellipse_a = 0
                ellipse_b = 0
            elif ctrl_line[1] == 0:
                ellipse_a = ctrl_line[0]
                ellipse_b = sideline[1]
            else:
                ellipse_a, ellipse_b = ctrl_line
        elif sideline[1] == 0:
            if ctrl_line[1] == 0:
                ellipse_a = 0
                ellipse_b = 0
            elif ctrl_line[0] == 0:
                ellipse_a = sideline[0]
                ellipse_b = ctrl_line[1]
            else:
                ellipse_a, ellipse_b = ctrl_line
        else:
            ellipse_a, ellipse_b = sideline
        if all((ellipse_a, ellipse_b)):
            prop_r0 = (ctrl_line[0] ** 2 / ellipse_a ** 2 + ctrl_line[1] ** 2 / ellipse_b ** 2) ** 0.5 / \
                      (ctrl_line_rotate[0] ** 2 / ellipse_a ** 2 + ctrl_line_rotate[1] ** 2 / ellipse_b ** 2) ** 0.5
            return Coordinate.get_points_on_line(((0, 0), ctrl_line_rotate), prop_r0)
        else:
            return ctrl_line

    @staticmethod
    def midpoint(points):
        """
        中点
        :param points:
        :return:
        """
        if len(points) == 2:
            return (np.array(points[0]) + np.array(points[1])) / 2
        midpoint = []
        midpoint_num = len(points) - 1
        for p_index in range(midpoint_num):
            p1 = np.array(points[p_index])
            p2 = np.array(points[p_index + 1])
            midpoint.append((p1 + p2) / 2)
        return midpoint
    
    @staticmethod
    def points_distance(points: list):
        """
        点之间的距离
        :param points:
        :return:
        """
        distance_list = []
        line_num = len(points) - 1
        if isinstance(points[0], (int, float)):
            axis_dimension = 1
        else:
            axis_dimension = len(points[0])
        for line_index in range(line_num):
            diff_sq = 0
            for axis in range(axis_dimension):
                if axis_dimension == 1:
                    diff_sq = abs(points[line_index] - points[line_index + 1])
                else:
                    diff_sq += (points[line_index][axis] - points[line_index + 1][axis]) ** 2
                    diff_sq = diff_sq ** 0.5
            distance_list.append(diff_sq)
        return distance_list

    @staticmethod
    def map_range(num, old, new):
        if (len(old) != 2) or (len(new) != 2):
            raise ValueError
        return (num-old[0])/(old[1]-old[0]) * (new[1]-new[0]) + new[0]


class Coordinate(object):
    def __init__(self):
        pass

    @staticmethod
    def rotate(point, degree):
        """
        将 点point 绕原点旋转 degree度
        :param point:       二维点坐标
        :param degree:      旋转度数, 角度制
        :return:            旋转后的点坐标
        """
        trans_matrix = np.array(((math.cos(degree), math.sin(degree), 0),
                                (math.sin(degree) * -1, math.cos(degree), 0),
                                (0, 0, 1)))
        point = np.array((point[0], point[1], 1))
        point_r = np.matmul(point, trans_matrix)
        return point_r[:2]

    @staticmethod
    def point_slope_angle(point, style='rad'):
        """
        求 点point与原点的连线 与 x轴正半轴 的夹角或斜率
        角度范围为(-180度, 180度], (-pi, pi]
        :param point:       二维点坐标
        :param style:       'deg':      返回角度制;
                            'rad':      返回弧度制
                            'slope':    返回斜率
        :return:            夹角或斜率
        """
        if point[0] == 0:
            if point[1] == 0:
                angle = 0
                slope = 0
            elif point[1] > 0:
                angle = math.pi / 2
                slope = None
            else:
                angle = math.pi / 2 * -1
                slope = None
            if style == 'slope':
                return slope
        else:
            slope = point[1] / point[0]
            if style == 'slope':
                return slope
            angle = math.atan(slope)
            if (point[0] < 0) and (point[1] >= 0):
                angle = math.pi + angle
            elif (point[0] < 0) and (point[1] < 0):
                angle = -1 * math.pi + angle
        if style == 'rad':
            return angle
        else:
            return angle * 180 / math.pi
    
    @staticmethod
    def get_lines_by_points(points):
        """
        返回一系列点顺次连接的线段
        :param points:      点序列
        :return:            线段列表, 线段形式: [point0, point1]
        """
        lines = []
        for i_f in range(len(points) - 1):
            lines.append([points[i_f], points[i_f + 1]])
        return lines

    def test(self):
        pass

    @staticmethod
    def get_points_on_line(line, prop):
        """
        获取线line上比例位置为prop的点坐标
        :param line:        线段, 形式: [point0, point1]
        :param prop:        比例值列表, 也可以是单个比例值
        :return:            点坐标列表或单个点
        """
        p1_np = np.array(line[0])
        p2_np = np.array(line[1])
        if isinstance(prop, (int, float)):
            p = p1_np + (p2_np - p1_np) * prop
        else:
            p = [p1_np + (p2_np - p1_np) * t for t in prop]
        return p

    @staticmethod
    def slope(points):
        """
        求一系列点顺次连接形成的线段与x轴正半轴的夹角的弧度
        :param points:      点列表
        :return:            弧度列表或单个弧度
        """
        if len(points[0]) != 2:
            print('输入坐标应只有两个维度')
            raise ValueError
        line_num = len(points) - 1
        slope_list = []
        for line_index in range(line_num):
            slope_list.append(Coordinate.point_slope_angle(np.array(points[line_index + 1]) - np.array(points[line_index])))
        if len(slope_list) == 1:
            return slope_list[0]
        else:
            return slope_list

    @staticmethod
    def xy_to_x_y(list_):
        """

        :param list_:
        :return:
        """
        result = []
        result_len = len(list_[0])
        for j in range(result_len):
            d = []
            for i_f in list_:
                d.append(i_f[j])
            result.append(d)
        return result
    
    @staticmethod
    def equal_distribute(points_x, points_y, equal_points_x):
        if equal_points_x[0] < points_x[0] or equal_points_x[-1] > points_x[-1]:
            raise ValueError
        # print(points_x, equal_points_x)
        start_index = 0
        equal_x_index = 0
        equal_points_y = []
        equal_points_num = len(equal_points_x)
        while True:
            left_x = points_x[start_index]
            right_x = points_x[start_index + 1]
            the_point_x = equal_points_x[equal_x_index]
            while not ((left_x <= the_point_x) and (the_point_x <= right_x)):
                start_index += 1
                left_x = points_x[start_index]
                right_x = points_x[start_index + 1]
            equal_points_y.append(points_y[start_index] + 
                                  (points_y[start_index+1] - points_y[start_index]) *
                                  ((the_point_x - left_x) / (right_x - left_x)))
            equal_x_index += 1
            if equal_x_index >= equal_points_num:
                break
        return equal_points_y
    
    @staticmethod
    def point_is_on_line(point, line):
        pass
        
    
def separate_coordinate(coordinate_list):
    axis_num = len(coordinate_list[0])
    result = []
    for which_axis in range(axis_num):
        axis = []
        for coordinate in coordinate_list:
            axis.append(coordinate[which_axis])
        result.append(axis)
    return result
