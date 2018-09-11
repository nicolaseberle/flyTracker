#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:37:36 2017

@author: neberle
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
import tracker_constant as const
import util
import csv

class MultipleObjectTracker(object):
    def __init__(self):
        self.tracks = []
        self.indiceTrack = 1
        self.indiceFrame = 0
        
    def __delete_duplicate_tracks(self):

        # check if tracks 'heads' are identical
        for i in np.arange(len(self.tracks)):
            track1 = self.tracks[i]
            for j in np.arange(len(self.tracks)):
                if j == i:
                    continue
                track2 = self.tracks[j]
                if util.check_tracks_equal(track1, track2):
                    print('duplicate found!')
                    # if so, delete shortest track
                    if track1.get_length() > track2.get_length():
                        track2.delete_me = True
                    else:
                        track1.delete_me = True
                        
        self.tracks = [t for t in self.tracks if t.delete_me is False]

    def update_tracks(self, current_plot):
        plots = []
        for _plot in current_plot:
                p = Plot()
                p.add_to_plot(_plot)
                plots.append(p)
        # if there are no tracks yet, all detections are new tracks
        if len(self.tracks) == 0:
            for _plot in current_plot:
                t = Track(self.indiceTrack)
                t.add_to_track(_plot)
                self.tracks.append(t)
                self.indiceTrack += 1
            return True
            
        dists = np.zeros(shape=(len(self.tracks), len(plots)))
        distsArea = np.zeros(shape=(len(self.tracks), len(plots)))
        if const.VERBOSE_LIGHT is True:
            print('nb tracks  ' + str(len(self.tracks)))
            print('nb plots ' + str(len(plots)))
        
        for i, track in enumerate(self.tracks):
            for j, _plot in enumerate(plots):
                
                dist = np.linalg.norm(_plot.plot-track.plot)
                distArea = np.linalg.norm(_plot.plot[0][3]-track.plot[0][3])
                if track.is_singular():
                    max_dist = const.MAX_PIXELS_DIST_TRACK_START
                else:
                    max_dist = const.MAX_PIXELS_DIST_TRACK
                    
                if dist > max_dist:
                    dist = 1e6  # set to arbitrarily high number
                    
                dists[i, j] = dist
                distsArea[i, j] = distArea
                
        if const.VERBOSE_FULL is True:        
            print(dists)
            print(distsArea )
        
        for t in self.tracks:
            t.has_match = False
            
        # assign all detections to tracks with munkres algorithm
        assigned_rows, assigned_cols = linear_sum_assignment(dists)
        for idx, row in enumerate(assigned_rows):
            col = assigned_cols[idx]
            # if track is assigned a detection with dist=1e6, discard that assignment
            if dists[row, col] != 1e6:
                self.tracks[row].has_match = True
                plots[col].has_match = True
                self.tracks[row].add_to_track(plots[col].plot)
                self.tracks[row].num_misses = 0

        # create new tracks from unassigned detections:
        for _plot in plots:
            #stop creating tracks after 30 frames
            if _plot.has_match is False and _plot.plot[0][3]>20 and self.indiceFrame<30:
                if const.VERBOSE_FULL is True: 
                    print("create new tracks from unassigned plot ")
                t = Track(self.indiceTrack)
                t.add_to_track(_plot.plot)
                t.num_misses = 0
                self.indiceTrack += 1
                self.tracks.append(t)
                
        # keep track of how many times a track has gone unassigned
        for t in self.tracks:
            if t.has_match is False:
                t.num_misses += 1
        # cleanup any duplicate tracks that have formed (TODO: how do they form?)
        self.__delete_duplicate_tracks()
        self.tracks = [t for t in self.tracks if (t.is_dead() is False and t.delete_me is False)]
        
        self.indiceFrame += 1
        return True

class Plot(object):
    
    def __init__(self):
        self.has_match = False
        self.plot = np.empty((0,1,4), dtype='float32')#pos_x,pos_y,orientation_rad,surface_pixel
        
    def add_to_plot(self,_plot):
        self.plot = _plot
        
class Track(object):
    def __init__(self,numPiste):
        
        self.has_match = False
        self.delete_me = False
        self.num_misses = 0
        self.nbplot = 0
        self.label = numPiste
        self.listPlot = np.empty((0,4), dtype='float32')
        self.listSpeed = np.empty((0,4), dtype='float32')
        self.plot = np.empty((0,1,4), dtype='float32')
        self.old_plot = np.empty((0,1,4), dtype='float32')
        self.name_foo = const.DIR_WORKSPACE +  'fly_res_out_' + str(numPiste) +".csv"
        self.init_once = False
        
        
    
        
    def createFoo(self):
        self.idx = 0 
        self.ofile = open(str(self.name_foo),"w")
        self.writer = csv.writer(self.ofile, delimiter=':')
        self.writer.writerow(["numFrame","X","Y","VX","VY"])
        
        
    def add_to_track(self,_plot):
        if( self.nbplot != 0 ):
            self.old_plot = self.plot
            
            
        self.plot = _plot
        self.listPlot = np.concatenate((self.listPlot,self.plot))
        
        if(self.nbplot !=0):
            self.compute_speed()
            
        if(self.nbplot > const.NBFRAME_BEFORE_RECORDING):
            if self.init_once is False:
                self.createFoo()
                self.init_once = True
            self.save_history()
            
        self.is_alive()
        
        self.nbplot += 1
        
    def save_history(self):
        self.writer.writerow([self.idx , const.CONV_PX2CM*self.plot[0][0],const.CONV_PX2CM*self.plot[0][1],self.speed[0][0], self.speed[0][1] ] )
        self.idx += 1
        
    def is_singular(self):
        if self.nbplot  == 1:
            return True
        else:
            return False
    def is_dead(self):
        if(self.num_misses > const.MAX_NUM_MISSES_TRACK):
            return True
        else:
            return False
        
    def is_alive(self):
        self.num_misses = 0
        
    def compute_speed(self):
        self.speed = (self.plot - self.old_plot)*const.DIAMETER_ROUND_BOX_CM/(0.04*const.DIAMETER_ROUND_BOX_PIXEL)
        self.listSpeed = np.concatenate((self.listSpeed,self.speed))
        
    def get_length(self):
        return self.nbplot
    
    def get_latest_bb(self):
        return self.plot;
    