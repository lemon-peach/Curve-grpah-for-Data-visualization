import configparser
import re
import os
from Bezier import *


class GetDataFromIni:
    def __init__(self):
        self.num_pattern = re.compile(r'^-?\d+(?:\.\d+)?$')
        self.int_pattern = re.compile(r'^-?\d+$')
        self.float_pattern = re.compile(r'^-?\d+(?:\.\d+)?$')
        self.l_brace_pattern = re.compile(r'{')
        self.r_brace_pattern = re.compile(r'}')
        self.l_paren_pattern = re.compile(r'\(')
        self.r_paren_pattern = re.compile(r'\)')

    @staticmethod
    def get_float(section, option, fallback):
        try:
            _value = float(config.get(section, option, fallback=fallback))
        except ValueError:
            return float(fallback)
        else:
            return _value

    @staticmethod
    def get_int(section, option, fallback):
        try:
            _value = int(config.get(section, option, fallback=fallback))
        except ValueError:
            return int(fallback)
        else:
            return _value

    @staticmethod
    def get_bool(section, option, fallback):
        _value = config.get(section, option, fallback=fallback)
        if _value in ('True', 'true'):
            return True
        else:
            return False

    def get_list(self, section, option, fallback):
        return self.get_list_from_str(config.get(section, option, fallback=fallback))

    def get_list_from_str(self, string: str):
        if len(self.l_paren_pattern.findall(string)) != len(self.r_paren_pattern.findall(string)):
            return None
        dict_d = 0
        list_d = 0
        new_string = ''
        _list = []
        string_len = len(string)
        for index, s in enumerate(string[1:-1]):

            if s == ' ':
                continue
            elif s == '(':
                list_d += 1
            elif s == ')':
                list_d -= 1
            elif s == '{':
                dict_d += 1
            elif s == '}':
                dict_d -= 1
            elif s == ',' and list_d == 0 and dict_d == 0:
                _list.append(self.get_param_from_str(new_string))
                new_string = ''
                continue
            new_string = ''.join((new_string, s)) if new_string else s
            if index == string_len - 3:
                _list.append(self.get_param_from_str(new_string))
        return _list

    def get_dict_from_str(self, string: str):
        if len(self.l_brace_pattern.findall(string)) != len(self.r_brace_pattern.findall(string)):
            return None
        _key = ''
        _value = ''
        key_value = 0
        dict_d = 0
        list_d = 0
        _dict = dict()
        string_len = len(string)
        for index, s in enumerate(string[1:-1]):
            if s == ' ':
                continue
            elif s == '{':
                dict_d += 1
            elif s == '}':
                dict_d -= 1
            elif s == '(':
                list_d += 1
            elif s == ')':
                list_d -= 1
            elif s == ':' and dict_d == 0:
                key_value = 1
                continue
            elif s == ',' and dict_d == 0 and list_d == 0:
                key_value = 0
                _dict[_key] = self.get_param_from_str(_value)
                _key = ''
                _value = ''
                continue
            if key_value == 0 and dict_d == 0:
                _key = ''.join((_key, s)) if _key else s
            elif key_value == 1:
                _value = ''.join((_value, s)) if _value else s
            if index == string_len - 3:
                _dict[_key] = self.get_param_from_str(_value)
                continue
        return _dict

    def get_param_from_str(self, string):
        if not string:
            return None
        if self.int_pattern.match(string):
            _value = int(string)
        elif self.float_pattern.match(string):
            _value = float(string)
        elif string[0] == '(' and string[-1] == ')':
            _value = self.get_list_from_str(string)
        elif string[0] == '{' and string[-1] == '}':
            _value = self.get_dict_from_str(string)
        elif string[0] == "\"" and string[-1] == "\"":
            _value = string[1:-1]
        elif string[0] == '\\':
            _value = string[1:]
        else:
            _value = string
        return _value

    @staticmethod
    def get_num_param(section, option, fallback):
        pass


date_code_dict = {'A': 0, 'D': 1, 'M': 2, 'Y': 3}
y_m_d_dict = {'Y': 10, 'M': 12, 'D': 31}

ini = GetDataFromIni()
config_path = r'./config/config.ini'
config = configparser.ConfigParser()
try:
    config.read(config_path, encoding='utf-8')
except configparser.MissingSectionHeaderError:
    config.read(config_path, encoding='utf-8-sig')

# [ico]
ico_type = config.get('ico', 'ico type', fallback='marker')
marker_style = ini.get_param_from_str(config.get('ico', 'marker style', fallback='.'))
ico_path_dict = dict(config.items('ico'))
ico_zoom = ini.get_float('ico', 'ico zoom', fallback='1')

# [figure]
# ----------------figure, axes---------------------
fig_size = ini.get_list('figure', 'fig size', fallback='(10, 5)')
if len(fig_size) != 2:
    fig_size = (10, 5)
if fig_size[0] > 15:
    fig_size = (15, fig_size[1])
if fig_size[1] > 8:
    fig_size = (fig_size[0], 8)
axes_rect = ini.get_list('figure', 'axes rect', fallback='(0.1, 0.1, 0.8, 0.8)')
if len(axes_rect) != 4:
    axes_rect = (0.1, 0.1, 0.8, 0.8)
axes_frame_on = ini.get_bool('figure', 'axes frame on', fallback='False')
right_space = ini.get_float('figure', 'right space', fallback='1.5')

# -----------------------line-------------------------
auto_fill = ini.get_bool('figure', 'auto fill', fallback='True')
bezier_smoothness = ini.get_int('figure', 'bezier smoothness', fallback='10')
bezier_auto_rotate = ini.get_bool('figure', 'bezier auto rotate', fallback='True')
bezier_auto_scale = ini.get_bool('figure', 'bezier auto scale', fallback='True')
bezier_scale_range = ini.get_list('figure', 'bezier scale range', fallback='(0.5, 0.9)')
if len(bezier_scale_range) != 2:
    bezier_scale_range = (0.5, 0.9)
point_div = 1 / ini.get_int('figure', 'point div', fallback='20')

# -----------------------grid-------------------------
y_grid_on = ini.get_bool('figure', 'y grid on', fallback='False')
x_grid_on = ini.get_bool('figure', 'x grid on', fallback='False')
y_grid_alpha = ini.get_float('figure', 'y grid alpha', fallback='0.2')
x_grid_alpha = ini.get_float('figure', 'y grid alpha', fallback='0.2')
y_grid_linestyle = ini.get_param_from_str(config.get('figure', 'y grid linestyle', fallback='--'))
if isinstance(y_grid_linestyle, list):
    y_grid_linestyle = tuple(y_grid_linestyle)
x_grid_linestyle = ini.get_param_from_str(config.get('figure', 'x grid linestyle', fallback='--'))
if isinstance(x_grid_linestyle, list):
    x_grid_linestyle = tuple(x_grid_linestyle)

# ----------------------axis---------------------
xlabel_prefix = config.get('figure', 'x label prefix', fallback='')
xlabel_suffix = config.get('figure', 'x label suffix', fallback='')
xlabel_rotation = ini.get_float('figure', 'x label rotation', fallback='0')
major_xlabel = config.get('figure', 'major x label', fallback='A')
major_xlabel_code = date_code_dict.get(major_xlabel, 0)
max_xsection_stage2 = ini.get_int('figure', 'max x section stage2', fallback='-1')
if max_xsection_stage2 == -1:
    max_xsection_stage2 = int(fig_size[0] / 2)
max_xsection_stage3 = ini.get_int('figure', 'max x section stage3', fallback='-1')
if max_xsection_stage3 == -1:
    max_xsection_stage3 = int(fig_size[0] / 1.5)

y_lim = ini.get_list('figure', 'y lim', fallback='(0, 0)')
y_label_format = config.get('figure', 'y label format', fallback='0')
ylabel_prefix = config.get('figure', 'y prefix', fallback='')
ylabel_suffix = config.get('figure', 'y suffix', fallback='')

# -------------------Annotation-------------------
x_unit = config.get('figure', 'x unit', fallback='')
x_unit_position = ini.get_list('figure', 'x unit position', fallback='(-1.0, -1.0)')
if len(x_unit_position) != 2:
    x_unit_position = (-1.0, -1.0)
y_unit = config.get('figure', 'y unit', fallback='')
y_unit_position = ini.get_list('figure', 'y unit position', fallback='(-1.0, -1.0)')
if len(y_unit_position) != 2:
    y_unit_position = (-1.0, -1.0)
legend_on = ini.get_bool('figure', 'legend on', fallback='False')
legend_color = ini.get_bool('figure', 'legend color', fallback='False')

# [line color]
line_color_dict = dict()
for key, value in config.items('line color'):
    line_color_dict[key] = ini.get_param_from_str(value)

# [video]
frame_div = 1 / ini.get_int('video', 'frame div', fallback='20')
video_fps = ini.get_int('video', 'fps', fallback='30')
video_type = config.get('video', 'video type', fallback='avi')
ffmpeg_path = config.get('video', 'ffmpeg path', fallback='')
synchro = ini.get_bool('video', 'synchro', fallback='False')
video_output_path = os.path.join(os.getcwd(), 'files/statistics.avi')

# [rcParams]
rcParams_dict = dict()
for key, value in config.items('rcParams'):
    rcParams_dict[key] = value if value[0] != '\\' else value[1:]

# [axes setattr]
_axes_setattr = dict(config.items('axes setattr'))
axes_setattr = dict()
for key, value in _axes_setattr.items():
    args = []
    kw = dict()
    axes_setattr[key] = {'args': [], 'kw': dict()}
    if value:
        for param in value.split(';'):
            param_split = param.split('=')
            param_split_len = len(param_split)
            if param_split_len == 1:
                args.append(ini.get_param_from_str(param_split[0]))
            elif param_split_len == 2:
                kw[param_split[0]] = ini.get_param_from_str(param_split[1])
            else:
                pass
    axes_setattr[key]['args'] = args
    axes_setattr[key]['kw'] = kw
