#Author: Ling Zhang
#ID: 23702691
#Course: CSC I6716 Computer Vision
#Project: Segmentation of 3D Cloud Points
#Description: Given a 3d point cloud, assign planar surfaces different colors.

import string, sys
from numpy import *
from numpy.linalg import *

class Image:

    def __init__(self,filename,skiplines):

        # open file
        lines = open(filename, "r").readlines()

        self.rows = rows = int(lines[0])
        self.cols = cols = int(lines[1])
        self.image = []
        color = array([-1,-1,-1])   #'color'; the initial [-1,-1,-1] means that it is unlabeled.

        n = skiplines     #start reading from lines here
        for i in range(0,rows):
            self.image.append([])
            for j in range(0,cols):
                l = lines[n].split(' ')
                coords = array([float(l[0]),float(l[1]),float(l[2])] )
                self.image[i].append( Point(coords,color) )
                n = n + 1

        self.image = array(self.image)

    def get_point(self,row,col):
        #returns a pair (array,list): the array is the x,y,z; the list is the color r,g,b.
        return self.image[row][col]

    def get_kxk_neighborhood(self,row,col,k):
        # return a k by k neighborhood points
        return array([      [ self.image[row-1][col-1].coords,   self.image[row-1][col].coords,   self.image[row-1][col+1].coords  ],
                            [ self.image[row][col-1].coords,     self.image[row][col].coords,     self.image[row][col+1].coords    ],
                            [ self.image[row+1][col-1].coords,   self.image[row+1][col].coords,   self.image[row+1][col+1].coords  ]  ])

class Point:

    def __init__(self,coords=array([0,0,0]),color=array([-1,-1,-1])):

        self.coords = coords
        self.color = color
        #equivalency class for use in sequential labeling algorithm...-1 means 'unlabeled'
        self.eclass_label = -1
        #planar, nonplanar, or unknown. unknown means it's not yet classified; nonplanar means it's on a discontinuity.
        self.type='unknown'
        self.is_boundary = False
        #  normal vector
        self.normal = array([0.,0.,0.])

# concate two class points to be one plane
class UnionFind:

    def __init__(self):
        self.leader = {}        #dictionary that given key=label, returns leader of label's eclass
        self.group = {}         #given a leader, return the points set it leads

    def add(self,a,b):

        leadera = self.leader.get(a)
        leaderb = self.leader.get(b)

        if leadera is not None:
            if leaderb is not None:
                if leadera == leaderb: return
                groupa = self.group[leadera]
                groupb = self.group[leaderb]
                if len(groupa) < len(groupb):
                    a, leadera, groupa, b, leaderb, groupb = b, leaderb, groupb, a, leadera, groupa
                # concate a and b
                groupa |= groupb
                del self.group[leaderb]
                for k in groupb:
                    self.leader[k] = leadera
            else:
                #if a's group isn't empty but b's group is, just stick b into a's group
                self.group[leadera].add(b)
                self.leader[b] = leadera
        else:
            if leaderb is not None:
                #if a's group is empty but b's group isn't, stick a into b's group
                self.group[leaderb].add(a)
                self.leader[a] = leaderb
            else:
                #both a's group and b's group were empty, so make a new group and stick them both into it with a as leader
                self.leader[a] = self.leader[b] = a
                self.group[a] = set([a, b])

    def make_new(self,a):
        self.leader[a] = a
        self.group[a] = set([a])

# determine if the p points are coplanar
def are_coplanar(P, p_thresh):

    k = len(P[0])
    centroid = array([0.,0.,0.])
    for i in range(0,k):
        for j in range(0,k):
            centroid = centroid + P[i][j]
    # centroid of the k by k matrix
    centroid = centroid/(k*k)
    # covariance matrix
    A = array([[0,0,0],[0,0,0],[0,0,0]])
    for i in range(0,k):
        for j in range(0,k):
            [p1,p2,p3] = P[i][j] - centroid
            # A = A + ([p1,p2,p3]^T times [p1,p2,p3])
            A = A + array([[p1],[p2],[p3]]) * array([p1,p2,p3])

    #Eigenvalues and corresponding eigenvectors of A.
    eigs = eigh(A)
    eigenvalues = eigs[0]
    eigenvectors = eigs[1]
    # smallest eigenvalue index
    min_eval_index = argmin(eigenvalues)
    # smallest eigenvalue
    min_eigenval = eigenvalues[min_eval_index]
    #The eigenvector associated with the smallest eigenvalue
    normal = eigenvectors[min_eval_index]
    #if the eigenvalue is small than threshold, then these points are coplanar so return True.
    if(min_eigenval <= p_thresh): return True
    else: return False

# 3 by 3 neighborhood points
k = 3
p_thresh = 0.0001
# filename = 'small_example.ptx'
# filename = 'big_example.ptx'
filename = 'big_example1.ptx'
skiplines = 10
# reading input into image array object
I = Image(filename,skiplines)

label=0
# seq labeling algorithm
E = UnionFind()

# Sequential labelling algorithm
for row in range(1,I.rows-1):
    for col in range(1,I.cols-1):
        #P is the 3x3 neighborhood centered about (row,col).
        P = I.get_kxk_neighborhood(row,col,k)

        if( are_coplanar(P, p_thresh) and I.get_point(row,col).coords.any() ):
            I.image[row][col].type='planar'

            #labels of N, W, and NW points. If these are zero or unlabeled, the value will be -1.
            p = I.get_point(row,col).eclass_label           #p the point need to label
            N = I.get_point(row-1,col).eclass_label         #up
            W = I.get_point(row,col-1).eclass_label         #left
            NW = I.get_point(row-1,col-1).eclass_label      #top left

            #Sequential labeling algorithm:
            if(NW is not -1):
                I.image[row][col].eclass_label = N
                E.add(N,W)
            elif(N is not -1): I.image[row][col].eclass_label = N
            elif(W is not -1): I.image[row][col].eclass_label = W
            else:
                label += 1
                I.image[row][col].eclass_label = label
                E.make_new(label)
        else:
            I.image[row][col].is_boundary=True
            I.image[row][col].type='nonplanar'

#assign a random color to each plane:
class_colors = {}
for e in E.group.keys():
    random.seed()
    color = array([ random.randint(0,255) , random.randint(0,255) , random.randint(0,255) ])
    class_colors[e] = color

for row in range(0,I.rows):
        for col in range(0,I.cols):
            if(I.get_point(row,col).coords.any() and I.get_point(row,col).eclass_label is not -1):      #if it's a planar nonzero, give it a color
                eclass = E.leader[ I.get_point(row,col).eclass_label ]
                I.image[row][col].color = class_colors[ eclass ]

#   output points to file
out_lines = []
out_lines.append(I.rows)
out_lines.append(I.cols)
line = ""
for row in range(1,I.rows-1):
        for col in range(1,I.cols-1):
            if( I.get_point(row,col).coords.any() ):
                line1 = " ".join([str(x) for x in I.get_point(row,col).coords])      #x,y,z
                line2 = " ".join([str(x) for x in I.get_point(row,col).color])      #color
                line = line1+' '+line2
                out_lines.append(line)
ofname = 'out.pts'
with open(ofname, mode='w') as fo:
        for l in out_lines:
            fo.write(str(l)+'\n')
