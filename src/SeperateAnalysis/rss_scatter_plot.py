#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 15:28:04 2022

@author: tabor
"""

import numpy as np
import matplotlib.pyplot as plt

font = 16


# set width of bar
barWidth = 0.25

Lanc1 = (0, 0.4470, 0.7410)
Lanc2 = (0.8500, 0.3250, 0.0980)
Lanc3 = (0.9290, 0.6940, 0.1250)


# set height of bar
results = []

font = font
plt.rcParams["font.size"] = font

fig, ax = plt.subplots(figsize=(8, 11.5))
results.append(
    [
        "Copper Pham et al.",
        [
            0.003967207703624147,
            0.007848725739662936,
            0.00803608784545308,
            0.006345099156226806,
            0.007690221021616354,
        ],
        [
            0.10757582947143132,
            0.2862570901027133,
            0.2840168440103608,
            0.19726181723751138,
            0.2832766846933571,
        ],
        [
            0.0034751467707221655,
            0.006260436771122923,
            0.006077968669753608,
            0.004913009331532641,
            0.005707891694898736,
        ],
        [
            0.07181770766028861,
            0.1835359878206817,
            0.181299910258741,
            0.11568504853666359,
            0.1772734158815557,
        ],
    ]
)
results.append(
    [
        "Copper known object  ",
        [
            0.0035512191948139614,
            0.006159223503907585,
            0.0035248635754527207,
            0.0038972157221661002,
            0.0036408949580156043,
        ],
        [
            0.03962517188458251,
            0.03329459421841978,
            0.040062976510780175,
            0.03700136839844116,
            0.04234206483943841,
        ],
        [
            0.0027854168372049228,
            0.005105715781319058,
            0.002742837100390038,
            0.0031608459972156623,
            0.0028890358561162585,
        ],
        [
            0.03535341131708276,
            0.028765640050266626,
            0.03572221606431845,
            0.03285793069850398,
            0.03754837819258625,
        ],
    ]
)
results.append(
    [
        "Copper adaptive",
        [
            0.005933484459280364,
            0.006681260642436528,
            0.006123364613463715,
            0.006688834303550449,
            0.0044788924139577135,
        ],
        [
            0.06328257077618843,
            0.066491627071539,
            0.06408369562880566,
            0.06563981740380241,
            0.06084930386483449,
        ],
        [
            0.004927277959729619,
            0.005393280296850038,
            0.004838996214825499,
            0.005174535909445928,
            0.003525314807842194,
        ],
        [
            0.05545554538314878,
            0.05790786764653703,
            0.055909265106322555,
            0.057381956115310245,
            0.053132563065239904,
        ],
    ]
)
results.append(
    [
        "Aluminum adaptive",
        [
            0.0024277606241863083,
            0.002959056336684879,
            0.003254531586321356,
            0.0025786281373198893,
            0.0039054417409109515,
        ],
        [
            0.0576466009695498,
            0.05773274464563875,
            0.04280860027255257,
            0.043548298953317834,
            0.044503256157619304,
        ],
        [
            0.002015500646757449,
            0.0024876647224172177,
            0.0025771801591647253,
            0.0022033644897165297,
            0.0030615785381218398,
        ],
        [
            0.05089007244153167,
            0.051229820915070653,
            0.03793075355926579,
            0.03866607286531067,
            0.03935152950303083,
        ],
    ]
)
results.append(
    [
        "Cuboid adaptive",
        [
            0.0024547971443333364,
            0.0025485428413359774,
            0.003552343641128518,
            0.002599817085030785,
            0.0025953765846709487,
        ],
        [
            0.06477380734266155,
            0.06406218031406614,
            0.06438000205176235,
            0.06553112488977841,
            0.06267261961823266,
        ],
        [
            0.0043017594134831865,
            0.004859612587745015,
            0.004702204051311663,
            0.005506528900436582,
            0.004671392084065191,
        ],
        [
            0.05131654965446603,
            0.05276449580047187,
            0.05545079333025884,
            0.052381300199757494,
            0.05069850090400468,
        ],
    ]
)
results.append(
    [
        "Cylinder adaptive",
        [
            0.0024277606241863083,
            0.002959056336684879,
            0.003254531586321356,
            0.0025786281373198893,
            0.0039054417409109515,
        ],
        [
            0.0576466009695498,
            0.05773274464563875,
            0.04280860027255257,
            0.043548298953317834,
            0.044503256157619304,
        ],
        [
            0.002171330865277957,
            0.002140012877731451,
            0.00292602389509785,
            0.0021564433633292214,
            0.002270424924409616,
        ],
        [
            0.05745579649191742,
            0.05688655677657309,
            0.05739422956837823,
            0.05836866377942978,
            0.05562359300973898,
        ],
    ]
)


names = [result[0] for result in results]
position_errors = np.array([np.median(result[3]) for result in results]) * 1000
position_errors_max = (
    np.array([np.max(result[3]) for result in results]) * 1000 - position_errors
)
position_errors_min = (
    position_errors - np.array([np.min(result[3]) for result in results]) * 1000
)

av_position_errors = np.array([result[3] for result in results]) * 1000
av_orientation_errors = np.array([result[4] for result in results])

# position_errors_std = np.array([result[2] for result in results]) * 1000
orientation_errors = np.array(
    [np.median(result[4]) for result in results]
)  # *180/np.pi
orientation_errors_max = (
    np.array([np.max(result[4]) for result in results]) - orientation_errors
)
orientation_errors_min = orientation_errors - np.array(
    [np.min(result[4]) for result in results]
)

# orientation_errors_std = np.array([result[4] for result in results]) #*180/np.pi

position_color = Lanc1
orientation_color = Lanc2
a = 0.0
edge_color = (0, 0, 0, 0)
cap_opacity = 1.0

# Set position of bar on X axis
br1 = list(range(len(names)))
br2 = [x + barWidth for x in br1]

# Make the plot
ax.bar(
    br1,
    position_errors,
    color=(position_color[0], position_color[1], position_color[2], a),
    width=barWidth,
    capsize=6,
    yerr=(position_errors_min, position_errors_max),
    edgecolor=edge_color,
    ecolor=(position_color[0], position_color[1], position_color[2], cap_opacity),
)
ax.scatter(br1 * 5, av_position_errors.T.flatten(), color=position_color)
ax.tick_params(axis="y", labelcolor=position_color)

ax2 = ax.twinx()
ax2.tick_params(axis="y", labelcolor=orientation_color)

ax2.scatter(br2 * 5, av_orientation_errors.T.flatten(), color=orientation_color)

ax2.bar(
    br2,
    orientation_errors,
    color=(orientation_color[0], orientation_color[1], orientation_color[2], a),
    width=barWidth,
    capsize=6,
    yerr=(orientation_errors_min, orientation_errors_max),
    edgecolor=edge_color,
    ecolor=(
        orientation_color[0],
        orientation_color[1],
        orientation_color[2],
        cap_opacity,
    ),
)


# Adding Xticks
# ax.set_xlabel('Experiment', fontweight ='bold', fontsize = 15)
ax.set_ylabel(
    "Position Error\n(mm)", fontweight="bold", fontsize=font, color=position_color
)
ax2.set_ylabel(
    "Orientation Error\n(radians)",
    fontweight="bold",
    fontsize=font,
    color=orientation_color,
)

# plt.xticks([r + barWidth for r in range(len(names))],
#       names, rotation=45)
ax.set_xticks([r + barWidth / 2 for r in range(len(names))])
ax.set_xticklabels(names, rotation=-45, ha="left", fontsize=font)

plt.show()
plt.tight_layout()
