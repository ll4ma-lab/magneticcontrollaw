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
        [0.0020667575767537396, 0.0020880214072635313, 0.002065432514759025],
        [0.17612697791853424, 0.13891845171811004, 0.17450872824627833]
    ]
)
results.append(
    [
        "Copper adaptive no prior",
        [0.008954000939082827, 0.007858609691721604, 0.007928749353677668],
        [0.14053674510060915, 0.21380164210790242, 0.11589210833286683]
    ]
)
results.append(
    [
        "Alum adaptive no prior",
        [0.0033948371927969663, 0.01512484879101682, 0.004144067485888561],
        [0.19631518719586386, 0.3385688026148828, 0.16355532397926],
    ]
)
results.append(
    [
        "Copper adaptive prior",
        [0.0032358634869498926, 0.0031393903934275936, 0.006126581811968283],
        [0.14740721837891907, 0.10988155114259628, 0.16221349074015595],
    ]
)
results.append(
    [
        "Alum adaptive prior",
        [0.0019067190093799325, 0.002830869341140554, 0.001562406400880411],
        [0.22735228062511195, 0.2114839338072715, 0.12270952101138954],
    ]
)
results.append(
    [
        "Copper adaptive inverse",
        [0.0011529285401641934, 0.001209218842829189, 0.0010743175376532338],
        [0.039498122765024964, 0.06331761434548187, 0.08380498589826933],
    ]
)
results.append(
    [
        "Alum adaptive inverse",
        [0.0030648790802203013, 0.0029801878646089006, 0.0028325917370792546],
        [0.13928064485428823, 0.13126056814411885, 0.1502124119730013],
    ]
)
results.append(
    [
        "Copper cuboid",
        [0.0011357314224207501, 0.0013265686911651789, 0.0011300231061376687],
        [0.061193393345963715, 0.033283597974041255, 0.07548176436405317],
    ]
)
results.append(
    [
        "Alum cylinder",
        [0.0010343007154203746, 0.001157092984392726, 0.0012185564264469706],
        [0.09929734126210962, 0.07779391947483728, 0.0853713453992245],
    ]
)
results.append(
    [
        "Copper scrap",
        [0.012915522404693063, 0.021036676971444818, 0.02529586803123518],
        [0.6767380138387169, 0.6085911714247695, 1.0277011930829052],
    ]
)
results.append(
    [
        "Alum structure",
        [0.004896629478957663, 0.0050989041031919485, 0.0059831287672878035],
        [0.8008181807619296, 0.7989514521158156, 1.0387137223529699],
    ]
)
results.append(
    [
        "Alum Extrusion",
        [0.0010277691844324992, 0.0011247551719146247, 0.001151178121461815],
        [0.10727158568532502, 0.08854179028007794, 0.10858888463424175],
    ]
)
results.append(
    [
        "Alum box",
        [0.0018781757349362533, 0.0009326445605163251, 0.0020357217022828784],
        [0.08482057248367007, 0.0857283457175815, 0.1574179736404268],
    ]
)
results.append(
    [
        "Alum box + small iron piece",
        [0.0018510472417794382, 0.0022233948036009805, 0.00141231223930121],
        [0.10354564998816423, 0.10212935004048285, 0.09159348190722584],
    ]
)
results.append(
    [
        "Alum box + big iron piece",
        [0.0034066412839554292, 0.0021015165394177304, 0.0030471128724119533],
        [0.16922741768435284, 0.09210807520742016, 0.12272452338630113],
    ]
)
results.append(
    [
        "Copper scrap modified",
        [0.0007929570924661587, 0.0006259101969913116, 0.0004134938684671559],
        [0.019671835612115718, 0.04409474238596917, 0.04057560214307449],
    ]
)
results.append(
    [
        "Alum structure modified",
        [0.0012833658649873429],
        [0.15155496260999268],
    ]
)
results.append(
    [
        "Alum box strongly initialized",
        [0.0004425410927559336],
        [0.009352275079602448],
    ]
)

mode = 'additional_experiments'

if mode == 'sphere':
    results = results[:8]
elif mode == 'nonsphere':
    results = results[8:16]
elif mode == 'additional_experiments':
    results = [results[index] for index in [10,11,13,16,17,18]]




names = [result[0] for result in results]
position_errors = np.array([np.median(result[1]) for result in results]) * 1000
position_errors_max = (
    np.array([np.max(result[1]) for result in results]) * 1000 - position_errors
)
position_errors_min = (
    position_errors - np.array([np.min(result[1]) for result in results]) * 1000
)

#av_position_errors = np.array([result[1] for result in results]) * 1000
#av_orientation_errors = np.array([result[2] for result in results])

# position_errors_std = np.array([result[2] for result in results]) * 1000
orientation_errors = np.array(
    [np.median(result[2]) for result in results]
)  # *180/np.pi
orientation_errors_max = (
    np.array([np.max(result[2]) for result in results]) - orientation_errors
)
orientation_errors_min = orientation_errors - np.array(
    [np.min(result[2]) for result in results]
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



ax.tick_params(axis="y", labelcolor=position_color)

ax2 = ax.twinx()
ax2.tick_params(axis="y", labelcolor=orientation_color)

for index,location in enumerate(br1):
    position_errors_local = np.array(results[index][1]) * 1000

    ax.scatter([location] * len(position_errors_local), position_errors_local, color=position_color)

for index,location in enumerate(br2):
    orientation_errors_local = np.array(results[index][2])

    ax2.scatter([location] * len(orientation_errors_local), orientation_errors_local, color=orientation_color)


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

fig.show()
fig.tight_layout()
fig.savefig(mode + '_quantitative.pdf')