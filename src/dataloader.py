# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 16:41:44 2020

@author: tabor
"""
import shelve


def save(filename, variable_dicts):
    my_shelf = shelve.open(filename, "n")  # 'n' for new

    for variable_dict in variable_dicts:

        for key in variable_dict.keys():
            try:
                my_shelf[key] = variable_dict[key]
            except:
                #
                # __builtins__, my_shelf, and imported modules can not be shelved.
                #
                print("ERROR shelving: {0}".format(key))
    my_shelf.close()


def load(filename, globals_in):
    my_shelf = shelve.open(filename, protocol=2)
    for key in my_shelf:
        try:
            globals_in[key] = my_shelf[key]
        except:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print("ERROR unshelving: {0}".format(key))
    my_shelf.close()
