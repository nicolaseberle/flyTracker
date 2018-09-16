#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 15:03:48 2018

@author: neberle
"""

import cv2
import numpy as np



class Fly(object):
    def __init__(self,position):
        self.Position = position;
        self.Angle = 0;
        self.Width = 10;
        self.Height = 10;
        self.MeanSquaredError = double.MaxValue;
    

    def Fit(self,img):
        MinorRadius = 10
        MajorRadius = 40
        # distance between the sampled points on the elipse circumference in degree
        angularResolution = 1
        #how many times did the fitted ellipse not change significantly?
        numConverged = 0
    
        #number of iterations for this fit
        numIterations = 0
    
        #repeat until the fitted ellipse does not change anymore, or the maximum number of iterations is reached
        for numIterations in range (100):
            if Converged == True:
                break
            else:
                #points on an ideal ellipse
                #CvPoint[] points;
                cv2.Ellipse2Poly(self.Position, (MajorRadius, MinorRadius), int(self.Angle), 0, 359, points,
                                angularResolution);
    
                #points on the edge of foregroudn to background, that are close to the elipse
                #CvPoint?[] edgePoints = new CvPoint?[points.Length];
    
                #remenber if the previous pixel in a given direction was foreground or background
                prevPixelWasForeground = np.empty((len(points)), dtype='bool')
    
                #when the first edge pixel is found, this value is updated
                firstEdgePixelOffset = 200;
    
                #from the center of the elipse towards the outside:
                for offset in range(-MajorRadius + 1,firstEdgePixelOffset + 20):
                    #draw an ellipse with the given offset
                    if offset > 0:
                        smallAxe = MinorRadius + offset
                    else:
                        smallAxe = MinorRadius + MinorRadius / MajorRadius * offset
                        
                    cv2.Ellipse2Poly(self.Position, (MajorRadius + offset,smallAxe ), int(self.Angle), 0,
                                    359, points, angularResolution)
                    for i in range(len(points)):
                    #for each angle
                        if edgePoints[i].HasValue: return; #edge for this angle already found
    
                        # check if the current pixel is foreground
                        if points[i].X < 0 or points[i].Y < 0 or points[i].X >= img.Cols or points[i].Y >= img.Rows:
                            foreground = False
                        else:
                            foreground = img.Get2D(points[i].Y, points[i].X).Val0 > 0
    
    
                        if prevPixelWasForeground[i] and foreground == False:                        
                            # found edge pixel!
                            edgePoints[i] = points[i];
                            # if this is the first edge pixel we found, remember its offset. the other pixels cannot be too far away, so we can stop searching soon
                            if offset < firstEdgePixelOffset and offset > 0:
                                firstEdgePixelOffset = offset
                        prevPixelWasForeground[i] = foreground;
                    
                
    
                # estimate the distance of each found edge pixel from the ideal elipse
                # this is a hack, since the actual equations for estimating point-ellipse distnaces are complicated
                cv2.Ellipse2Poly(self.Position, (MajorRadius, MinorRadius), int(self.Angle), 0, 360,
                                points, angularResolution)
                pointswithDistance = edgePoints.Select((p, i) => p.HasValue ? new EllipsePoint(p.Value, points[i], this.Position) : null)
                              .Where(p => p != null).ToList();
    
                if pointswithDistance.Count == 0:
                    Console.WriteLine("no points found! should never happen! ");
                    break;
    
                # throw away all outliers that are too far outside the current ellipse
                double medianSignedDistance = pointswithDistance.OrderBy(p => p.SignedDistance).ElementAt(pointswithDistance.Count / 2).SignedDistance;
                var goodPoints = pointswithDistance.Where(p => p.SignedDistance < medianSignedDistance + 15).ToList();
    
                # do a sort of ransack fit with the inlier points to find a new better ellipse
                CvBox2D bestfit = ellipseRansack(goodPoints);
    
                # check if the fit has converged
                if np.Abs(this.Angle - bestfit.Angle) < 3 and # angle has not changed much (<3°)
                    np.Abs(this.Position.X - bestfit.Center.X) < 3 and # position has not changed much (<3 pixel)
                    np.Abs(this.Position.Y - bestfit.Center.Y) < 3:                
                    numConverged = numConverged + 1;
                else:
                    numConverged = 0;
                
    
                if numConverged > 2:
                    self.Converged = true;
                
    
                #Console.WriteLine("Iteration {0}, delta {1:0.000} {2:0.000} {3:0.000}    {4:0.000}-{5:0.000} {6:0.000}-{7:0.000} {8:0.000}-{9:0.000}",
                #  numIterations, Math.Abs(this.Angle - bestfit.Angle), Math.Abs(this.Position.X - bestfit.Center.X), Math.Abs(this.Position.Y - bestfit.Center.Y), this.Angle, bestfit.Angle, this.Position.X, bestfit.Center.X, this.Position.Y, bestfit.Center.Y);
    
                double msr = goodPoints.Sum(p => p.Distance * p.Distance) / goodPoints.Count;
    
                # for drawing the polygon, filter the edge points more strongly
                if goodPoints.Count(p => p.SignedDistance < 5) > goodPoints.Count / 2:
                    goodPoints = goodPoints.Where(p => p.SignedDistance < 5).ToList()
                cutoff = goodPoints.Select(p => p.Distance).OrderBy(d => d).ElementAt(goodPoints.Count * 9 / 10);
                goodPoints = goodPoints.Where(p => p.SignedDistance <= cutoff + 1).ToList()
    
                numCertainEdgePoints = goodPoints.Count(p => p.SignedDistance > -2)
                self.CircumferenceRatio = numCertainEdgePoints * 1.0 / points.Count()
    
                self.Angle = bestfit.Angle;
                self.Position = bestfit.Center;
                self.Width = bestfit.Size.Width;
                self.Height = bestfit.Size.Height;
                self.EdgePoints = goodPoints;
                self.MeanSquaredError = msr;
    
            }
            self.NumIterations = numIterations;
            #Console.WriteLine("Grain found after {0,3} iterations, size={1,3:0.}x{2,3:0.}   pixel={3,5}    edgePoints={4,3}   msr={5,2:0.00000}", numIterations, this.Width,
            #                        this.Height, this.NumPixel, this.EdgePoints.Count, this.MeanSquaredError);
        }