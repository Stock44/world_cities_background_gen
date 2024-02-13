import math
import random
from math import floor, cos, sin, pi

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from catppuccin import Flavour

MOCHA = Flavour.mocha()

# country name or None for world
# proper names
COUNTRY: str | None = None

# list of colors for cities above second threshold
COLORS = [
    MOCHA.red.hex,
    MOCHA.blue.hex,
    MOCHA.pink.hex,
    MOCHA.yellow.hex,
    MOCHA.green.hex,
    MOCHA.teal.hex,
    MOCHA.peach.hex,
]

FOREGROUND = MOCHA.surface0.hex  # city color

BACKGROUND = MOCHA.base.hex  # background color

RESOLUTION = (3456, 2160)

MARGINS = [196, 512, 196, 512]  # top right bottom left

GRID_SPACING = 16  # space between city dots

GRID_CELL_SIZE = 16  # size of city dots

SUPERSAMPLE_SCALE = 2  # scale to use for supersample antialiasing

COLOR_THRESHOLD = 5000000  # population threshold to color a city

MIN_POP_THRESHOLD = 100000  # population threshold to show a city in the map

ROTATION_ANGLE: float = 0 * pi / 180  # should be in radians


def rotate_coords(row):
    row.lng = row.lng * cos(ROTATION_ANGLE) + row.lat * sin(ROTATION_ANGLE)
    row.lat = - row.lng * sin(ROTATION_ANGLE) + row.lat * cos(ROTATION_ANGLE)

    return row


def main():
    cities = pd.read_csv('worldcities.csv')
    cities = cities.where(cities['population'] > MIN_POP_THRESHOLD)
    if COUNTRY is not None:
        cities = cities.where(cities['country'] == COUNTRY)

    cities = cities.apply(rotate_coords, 1)

    resolution = [RESOLUTION[0] * SUPERSAMPLE_SCALE, RESOLUTION[1] * SUPERSAMPLE_SCALE]
    margins = [MARGINS[0] * SUPERSAMPLE_SCALE, MARGINS[1] * SUPERSAMPLE_SCALE, MARGINS[2] * SUPERSAMPLE_SCALE,
               MARGINS[3] * SUPERSAMPLE_SCALE]

    cell_spacing = GRID_SPACING * SUPERSAMPLE_SCALE
    cell_size = GRID_CELL_SIZE * SUPERSAMPLE_SCALE

    big_image = Image.new('RGB', resolution, "#{}".format(BACKGROUND))

    lon_bounds = (cities['lng'].min(), cities['lng'].max())
    lon_range = lon_bounds[1] - lon_bounds[0]
    lat_bounds = (cities['lat'].min(), cities['lat'].max())
    lat_range = lat_bounds[1] - lat_bounds[0]

    area_width = resolution[0] - margins[1] - margins[3]
    area_height = resolution[1] - margins[0] - margins[2]

    country_aspect_ratio = lon_range / lat_range
    area_aspect_ratio = area_width / area_height

    if country_aspect_ratio > area_aspect_ratio:
        grid_width = area_width
        grid_height = grid_width / country_aspect_ratio

        extra_margin = (area_height - grid_height) // 2
        margins[0] += extra_margin
        margins[2] += extra_margin
    else:
        grid_height = area_height
        grid_width = grid_height * country_aspect_ratio

        extra_margin = (area_width - grid_width) // 2
        margins[1] += extra_margin
        margins[3] += extra_margin

    v_cells = floor(grid_height / (cell_size + cell_spacing))
    h_cells = floor(grid_width / (cell_size + cell_spacing))
    grid = np.zeros((h_cells, v_cells))

    for idx, row in cities.iterrows():
        if not isinstance(row['city'], str) and math.isnan(row['city']): continue
        lim_lon = row['lng'] - lon_bounds[0]
        lim_lat = -(row['lat'] - lat_bounds[0])
        h_pos = lim_lon / lon_range
        v_pos = lim_lat / lat_range
        x = h_pos * (h_cells - 1)
        y = v_pos * (v_cells - 1)

        x = floor(x)
        y = floor(y)

        grid[x][y] += row['population']
        # if row['population'] > grid[x][y]:
        #     grid[x][y] = row['population']

    d = ImageDraw.Draw(big_image)

    for x in range(h_cells):
        for y in range(v_cells):
            population = grid[x][y]

            if population > COLOR_THRESHOLD:
                color = random.choice(COLORS)
            elif population > MIN_POP_THRESHOLD:
                color = FOREGROUND
            else:
                continue

            corner1 = (margins[1] + x * (cell_size + cell_spacing),
                       margins[0] + y * (cell_size + cell_spacing))
            corner2 = (corner1[0] + cell_size, corner1[1] + cell_size)

            d.ellipse([corner1, corner2], fill="#{}".format(color))

    output_im = big_image.resize(RESOLUTION, Image.LANCZOS)

    output_im.save('%s-%dx%d.png' % (COUNTRY.lower() if COUNTRY is not None else 'world', RESOLUTION[0], RESOLUTION[1]),
                   'png')


if __name__ == '__main__':
    main()
