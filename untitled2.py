# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:19:47 2021

@author: Griffin Tabor
"""
import matplotlib.text as mtext
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, r'\quad \underline{' + orig_handle + '}', usetex=True, **self.text_props)
        handlebox.add_artist(title)
        return title

fig, ax = plt.subplots()

ds = [1,2,3]
dc = [1.1, 1.9, 3.2]
asim = [1.5, 2.2, 3.1]
ac = [1.6, 2.15, 3.1]

categories = ['x', 'y']

p1, = ax.plot(ds, 'ko', label='x')
p2, = ax.plot(dc, 'k:', label='y')
p3, = ax.plot(asim, 'b+', label='x ')
p4, = ax.plot(ac, 'b-', label='y ')


# Legend
handles, labels = ax.get_legend_handles_labels()

handles.insert(0, 'Position')
labels.insert(0, '')

handles.insert(3, 'Orientation')
labels.insert(3, '')

leg0 = ax.legend(handles, labels, handler_map={str: LegendTitle({'fontsize': 14})}, loc='lower left')


p5 = ''
p7 = ''

pos = 'Position'
orien = 'Orientation'
leg = ax.legend([p5, p1, p2, p5, p3, p4],
              [orien] + categories + [pos] + categories,
              loc=2, ncol=2,handler_map={str: LegendTitle({'fontsize': 14})}) # Two columns, vertical group labels

leg4 = ax.legend([p5, p7, p5, p7, p1, p2, p3, p4],
              [orien, '', pos, ''] + categories + categories,
               loc=4, ncol=2,handler_map={str: LegendTitle({'fontsize': 14})}) # Two columns, horizontal group labels

plt.gca().add_artist(leg4)
plt.gca().add_artist(leg)



# Get the bounding box of the original legend
bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)

# Change to location of the legend. 
xOffset = 0.1
bb.x0 += xOffset
#bb.x1 += xOffset
leg.set_bbox_to_anchor(bb, transform = ax.transAxes)

plt.show()




# Plots





