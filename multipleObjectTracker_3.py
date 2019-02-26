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
import cv2
import logging

class MultipleObjectTracker(object):
    def __init__(self):
        self.tracks = []
        self.indiceTrack = 0
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
                    if const.VERBOSE_FULL is True:
                        print('duplicate found!')
                    # if so, delete shortest track
                    if track1.get_length() > track2.get_length():
                        track2.delete_me = True
                        if const.VERBOSE_FULL is True:
                            print("delete track : " + str(track2.label))
                    else:
                        track1.delete_me = True
                        if const.VERBOSE_FULL is True:
                            print("delete track : " + str(track1.label))

        self.tracks = [t for t in self.tracks if t.delete_me is False]

    def update_tracks(self, current_plot, current_date):
        plots = []
        print(current_plot.shape)
        for _plot in current_plot:
                p = Plot()
                p.add_to_plot(_plot,current_date)
                plots.append(p)
        # if there are no tracks yet, all detections are new tracks
        if len(self.tracks) == 0:
            for _plot in current_plot:
                t = Track(self.indiceTrack)
                p_ = Plot()
                p_.add_to_plot(_plot,current_date)
                t.add_to_track(p_)
                self.tracks.append(t)
                self.indiceTrack += 1
            return True
        #init of dist : the distance matrix between current_plot and the plots of tracks
        dists = np.zeros(shape=(len(self.tracks), len(plots)))
        distsArea = np.zeros(shape=(len(self.tracks), len(plots)))
        if const.VERBOSE_LIGHT is True:
            print('nb tracks  ' + str(len(self.tracks)))
            print('nb plots ' + str(len(plots)))

        #compute the distance matrix between current_plot and the plots of tracks
        for i, track in enumerate(self.tracks):
            for j, _plot in enumerate(plots):

                dist = np.linalg.norm(_plot.plot[0][:2]-track.plot[0][:2])#ONLY X,Y NO AREA, NO ANGLE

                distArea = np.linalg.norm(_plot.plot[0][3]-track.plot[0][3])#ONLY AREA
                if track.is_singular():
                    max_dist = track.roi_of_search#const.MAX_PIXELS_DIST_TRACK_START
                else:
                    max_dist = track.roi_of_search#const.MAX_PIXELS_DIST_TRACK

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
                self.tracks[row].add_to_track(plots[col])
                self.tracks[row].num_misses = 0
                self.tracks[row].area = plots[col].plot[0][3]
                self.tracks[row].flag_cluster = False
                self.tracks[row].updateStatus()


        # create new tracks from unassigned detections:
        for _plot in plots:

            if _plot.has_match is False and _plot.plot[0][3]>20 and self.indiceFrame < 100 :
                if const.VERBOSE_LIGHT is True:
                    print("create new tracks from unassigned plot ")

                new_track = Track(self.indiceTrack)
                new_track.add_to_track(_plot)
                self.tracks.append(new_track)
                self.indiceTrack += 1

        #manage the cluster with unassigned tracks inside it
        for t in self.tracks:
            if t.has_match is True:
                liste_track_no_assigned = []
                liste_track_no_assigned.append(t)
                #we look for unassigned tracks around a matched track
                for _t in self.tracks:
                    ellipse = t.plot[0][4:9]
                    if  t.label!=_t.label and _t.has_match is False \
                                    and self.indiceFrame>1 and t.is_singular() == False \
                                        and (t.plot[0][3] > 1.3*t.old_plot[0][3] or t.flag_cluster == True or _t.flag_cluster==True):
                        if util.testPtsInEllipse(ellipse,_t.plot,_t.speed,const.FPS) is True:
                            liste_track_no_assigned.append(_t)
                            if const.VERBOSE_FULL is True:
                                print('autour de ' + str(t.label) +  'track ' + str(_t.label) + ' non assigné')

                #some tracks have been found
                n = len(liste_track_no_assigned)
                if n > 1:#>1 because we add at the beginning the matched track
                    t.flag_cluster = True
# a relire
                    for i__ in range(n):
                        if liste_track_no_assigned[i__].nbplot>2 :
                            liste_track_no_assigned[0].plot[0][:2] = liste_track_no_assigned[0].old_plot[0][:2]
                            print("INIT Cluster Track",liste_track_no_assigned[i__].old_plot[0][:2],liste_track_no_assigned[i__].plot[0][:2])
#
                    #we load the n centroids from kmean of the cluster track. n = number of unassigned tracks in the cluster
                    p = np.empty(shape=(n,3), dtype='float32')
                    #offset pour récupérer les centres issus du Kmean
                    offset = 6
                    for iteration in range(1,n):
                        offset = offset + 3*(n-iteration)
                    for k in range(n):
                        p[k][:3] = t.plot[0][offset+k*3:offset+(k+1)*3]
                    dists = np.zeros(shape=(n, n))

                    for j in range(n):
                        for i,_t in enumerate(liste_track_no_assigned):

                            dt = 1/const.FPS
                            #attention old_plot n'existe pas forcément à la deuxième itération
                            estime_current_plot = _t.old_plot[0][:2]+_t.speed[0][:2]*dt
                            dists[i,j] = np.linalg.norm(p[j][:2]-estime_current_plot)
                            #dists[i,j] = np.linalg.norm(p[j][:2]-_t.old_plot[0][:2])
                            if const.VERBOSE_FULL is True:
                                print(i,j,dists[i,j])

                    assigned_rows, assigned_cols = linear_sum_assignment(dists)

                    for idx, row in enumerate(assigned_rows):

                        col = assigned_cols[idx]
                        p_ = np.copy(liste_track_no_assigned[row].plot)
                        p_[0][:2] = p[col][:2]
                        p_[0][3] = p[col][2]
                        p__ = Plot()
                        p__.add_to_plot(p_,current_date)

                        if const.VERBOSE_FULL is True:
                            print(row,col,' track ' + str(liste_track_no_assigned[row].label) + ' ' + str(liste_track_no_assigned[row].old_plot[0][0]) + ','+ str(liste_track_no_assigned[row].old_plot[0][1]) + ' -> ' + str(p_[0][0]) + ',' + str(p_[0][1]))
                        plots.append(p_)
                        liste_track_no_assigned[row].has_match = True

                        #liste_track_no_assigned[row].plot = liste_track_no_assigned[row].old_plot
                        liste_track_no_assigned[row].num_misses = 0
                        liste_track_no_assigned[row].flag_cluster = True
                        liste_track_no_assigned[row].add_to_track(p__)
                        liste_track_no_assigned[row].updateStatus()
                        liste_track_no_assigned[row].area = p_[0][3]

        # keep track of how many times a track has gone unassigned
        for t in self.tracks:
            if t.has_match is False and self.indiceFrame>1 and t.is_singular() == False:
                #INCREASE THE ROI OF SEARCH THE PLOT
                if t.roi_of_search > const.MAX_PIXELS_DIST_TRACK*3:
                    t.roi_of_search = const.MAX_PIXELS_DIST_TRACK*3
                else:
                    t.roi_of_search = t.roi_of_search + 5

        # cleanup any duplicate tracks that have formed (TODO: how do they form?)
        self.__delete_duplicate_tracks()
        self.tracks = [t for t in self.tracks if (t.is_dead() is False and t.delete_me is False)]

        self.indiceFrame += 1
        return True

class Plot(object):

    def __init__(self):
        self.has_match = False
        self.plot = np.empty((0,1,51), dtype='float32')#pos_x,pos_y,orientation_rad,surface_pixel
        self.date = None

    def add_to_plot(self, _plot, _date):
        self.plot = _plot
        self.date = _date

class Track(object):
    def __init__(self,numPiste):

        self.has_match = False
        self.delete_me = False
        self.num_misses = 0
        self.nbplot = 0
        self.label = numPiste
        #self.listPlot = np.empty((0,51), dtype='float32')
        #self.listSpeed = np.empty((0,51), dtype='float32')
        self.plot = np.empty((0,1,51), dtype='float32')
        self.speed = np.zeros((1,51), dtype='float32')
        #self.old_plot = np.empty((0,1,51), dtype='float32')
        self.name_foo = const.DIR_WORKSPACE +  'fly_res_out_' + str(numPiste) +".csv"
        self.init_once = False
        self.roi_of_search = const.MAX_PIXELS_DIST_TRACK
        self.area = -1#NOT COMPUTE DURING INIT
        self.flag_cluster = False
        self.flag_touch = 0
        self.old_date = None
        self.current_date = None


    def createFoo(self):
        self.idx = 0
        self.ofile = open(str(self.name_foo),"w")
        self.writer = csv.writer(self.ofile, delimiter=':')
        self.writer.writerow(["Date(ms)","numFrame","X (in pixel)","Y (in pixel)","VX (in pixel/s)","VY (in pixel/s)",'T/A'])#Alone or Touch


    def add_to_track(self,_plot):
        if self.nbplot > 1 :
            self.old_old_plot = self.old_plot
        if self.nbplot > 0 :
            self.old_plot = self.plot
            self.old_date = self.current_date

        self.plot = _plot.plot
        self.current_date = _plot.date
        #self.listPlot = np.concatenate((self.listPlot,self.plot))

        if(self.nbplot > 0):
            self.compute_speed()


        if(self.nbplot > const.NBFRAME_BEFORE_RECORDING):
            if self.init_once is False:
                #self.createFoo()
                self.init_once = True
            #self.save_history()

        self.is_alive()

        self.nbplot += 1

    def save_history(self):
        self.writer.writerow([self.idx , self.plot[0][0],self.plot[0][1],self.speed[0][0], self.speed[0][1],self.flag_touch ] )
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
        self.roi_of_search = const.MAX_PIXELS_DIST_TRACK

    def compute_speed(self):
        if self.nbplot  > 1 and self.old_date != self.current_date:
            dt = (self.current_date - self.old_date)/1000 # millisecond -> second conversion
        else:
            dt = 1/const.FPS # millisecond -> second conversion
            #self.listSpeed = np.concatenate((self.listSpeed,self.speed))

        self.speed = (self.plot - self.old_plot)/dt

    def get_length(self):
        return self.nbplot

    def get_latest_bb(self):
        return self.plot;

    def updateStatus(self):
        if self.flag_cluster is True:
            self.flag_touch = 1
        else:
            self.flag_touch = 0
