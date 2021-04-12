'''
parse the sar map from excel files
requirements:
pip install xlrd
'''

import xlrd
import numpy as np
from pathlib import Path
import json
import copy

RESOURCES_DIR = (Path(__file__).parent / './envs/resources').resolve()
map_name = 'Falcon_difficult'  # 'Falcon_easy' or 'Falcon_medium' or 'Falcon_difficult'
map_path = Path(RESOURCES_DIR, map_name + '.xls')
map_info = Path(RESOURCES_DIR, 'Falcon_map.json')
add_doors = False  # if consider doors

book = xlrd.open_workbook(map_path, formatting_info=True)
sheets = book.sheet_names()
print("sheets are:", sheets)
minigrid_map = np.zeros((91, 51), dtype=np.int32) + 1

for index, sh in enumerate(sheets):
    sheet = book.sheet_by_index(index)
    if sheet.name not in ['Falcon 5 Low', 'Falcon 5 Med', 'Falcon 5 High']:
        continue
    # print("Sheet:", sheet.name)
    rows, cols = sheet.nrows, sheet.ncols
    # print("Number of rows: %s   Number of cols: %s" % (rows, cols))
    rows, cols = 94, 51
    for row in range(3, rows):
        for col in range(cols):
            # print("row, col is:", row+1, col+1,)
            thecell = sheet.cell(row, col)
            # could get 'dump', 'value', 'xf_index'
            if thecell.value == 'Y':
                minigrid_map[row - 3, col] = 81
            elif thecell.value == 'G':
                minigrid_map[row - 3, col] = 82

            xfx = sheet.cell_xf_index(row, col)
            xf = book.xf_list[xfx]
            bgx = xf.background.pattern_colour_index
            if bgx != 64:
                if bgx == 22:
                    minigrid_map[row - 3, col] = 4  # walls
                elif bgx == 8:
                    minigrid_map[row - 3, col] = 5  # blockages
                elif bgx == 30:
                    minigrid_map[row - 3, col] = 255  # boxes
                elif bgx == 52:
                    minigrid_map[row - 3, col] = 2  # openings

    minigrid_map[0, :] = minigrid_map[:, 0] = minigrid_map[:, -1] = 4  # add three lines of wall

minigrid_map = np.rot90(minigrid_map, 3)

if add_doors:
    with open(map_info, 'r') as load_map_json:
        map_json = json.load(load_map_json)

    for key, value in map_json.items():
        if key == 'connections':
            for i in range(len(map_json[key])):
                if map_json[key][i]['type'] == 'door':
                    door_y = map_json[key][i]['bounds']['coordinates'][0]['x'] + 2110
                    door_x = map_json[key][i]['bounds']['coordinates'][0]['z'] - 142
                    minigrid_map[door_x, door_y] = 2
                elif map_json[key][i]['type'] == 'double_door':
                    lt_x = map_json[key][i]['bounds']['coordinates'][0]['x']
                    lt_z = map_json[key][i]['bounds']['coordinates'][0]['z']
                    br_x = map_json[key][i]['bounds']['coordinates'][1]['x']
                    br_z = map_json[key][i]['bounds']['coordinates'][1]['z']
                    if br_x - lt_x > 1:
                        door_y = lt_x + 2110
                        door_x = lt_z - 142
                        minigrid_map[door_x, door_y] = 2
                        door_y = lt_x + 2111
                        door_x = lt_z - 142
                        minigrid_map[door_x, door_y] = 2
                    else:
                        door_y = lt_x + 2110
                        door_x = lt_z - 142
                        minigrid_map[door_x, door_y] = 2
                        door_y = lt_x + 2110
                        door_x = lt_z - 141
                        minigrid_map[door_x, door_y] = 2

raw_map_state = copy.deepcopy(minigrid_map)
raw_map_state[raw_map_state == 2] = 4
raw_map_state[raw_map_state == 5] = 1
raw_map_state[raw_map_state == 81] = 1
raw_map_state[raw_map_state == 82] = 1

minigrid_map[minigrid_map == 2] = 1
minigrid_map[minigrid_map == 5] = 4

saved_path = Path(RESOURCES_DIR, map_name + '.npy')
np.save(saved_path, minigrid_map)

saved_path = Path(RESOURCES_DIR, 'raw_map_state.npy')
np.save(saved_path, raw_map_state)
