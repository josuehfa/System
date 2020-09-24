#!/usr/bin/ebv python
# coding=utf-8


import sys
from pprint import pprint as pp
from collections import namedtuple

""" Based om http://rosettacode.org/wiki/Ray-casting_algorithm#Python
"""

Pt = namedtuple('Pt', 'x, y')               # Point
Edge = namedtuple('Edge', 'a, b')           # Polygon edge from a to b
Poly = namedtuple('Poly', 'name, edges')    # Polygon


def rayintersectseg(p, edge):
    ''' takes a point p=Pt() and an edge of two endpoints a,b=Pt() of a line segment returns boolean
    '''
    _eps = 0.00001
    _huge = sys.float_info.max
    _tiny = sys.float_info.min 
    a,b = edge
    if a.y > b.y:
        a,b = b,a
    if p.y == a.y or p.y == b.y:
        p = Pt(p.x, p.y + _eps)

    intersect = False

    if (p.y > b.y or p.y < a.y) or (
        p.x > max(a.x, b.x)):
        return False

    if p.x < min(a.x, b.x):
        intersect = True
    else:
        if abs(a.x - b.x) > _tiny:
            m_red = (b.y - a.y) / float(b.x - a.x)
        else:
            m_red = _huge
        if abs(a.x - p.x) > _tiny:
            m_blue = (p.y - a.y) / float(p.x - a.x)
        else:
            m_blue = _huge
        intersect = m_blue >= m_red
    return intersect

def _odd( x): return x%2 == 1

def ispointinside( p, poly):
    ln = len(poly)
    return _odd(sum(rayintersectseg(p, edge)
                    for edge in poly.edges ))

#def polypp(poly):
#    print ("\n  Polygon(name='%s', edges=(" + poly.name)
#    print ('   ' + ',\n    '.join(str(e) for e in poly.edges) + '\n    ))')

# if __name__ == '__main__':
#     polys = [
#         Poly(name='avwilliams', edges=(
#             Edge(a=Pt(x=38.9913160, y=-76.937079), b=Pt(x=38.991333, y=-76.936119)),
#             Edge(a=Pt(x=38.991333, y=-76.936119), b=Pt(x=38.990287, y=-76.936108)),
#             Edge(a=Pt(x=38.990287, y=-76.936108), b=Pt(x=38.990278, y=-76.937057)),
#             Edge(a=Pt(x=38.990278, y=-76.937057), b=Pt(x=38.990495,y=-76.937052)),
#             Edge(a=Pt(x=38.990495,y=-76.937052), b=Pt(x=38.990499,y=-76.936424)),
#             Edge(a=Pt(x=38.990499,y=-76.936424), b=Pt(x=38.991091,y=-76.93643)),
#             Edge(a=Pt(x=38.991091,y=-76.93643), b=Pt(x=38.991104,y=-76.937079)),
#             Edge(a=Pt(x=38.991104,y=-76.937079), b=Pt(x=38.9913160, y=-76.937079))
#             )),
#     ]  

#     #if len(sys.argv) != 3:
#     #    print ("Incorrect number of arguments.  Please submit a lat and a long....")

#     #userpoint = (Pt(x=float(sys.argv[1]), y=float(sys.argv[2])))

#     testpoints = (Pt(x=0, y=0), Pt(x=-38.990842, y=-76.93625),
#                   Pt(x=38.9021466, y=-77), Pt(x=0, y=5),
#                   Pt(x=10, y=5), Pt(x=8, y=5),
#                   Pt(x=10, y=10))
 
#     #print "\n TESTING WHETHER POINTS ARE WITHIN POLYGONS"
#     inside = False
#     for poly in polys:
        
#         #polypp(poly)
#         #print ('   ' + '\t'.join(str(p), str(ispointinside(p, poly))))
#         #                       for p in testpoints[:3])
#         if ispointinside(testpoints[1], poly):
#             inside = True
#             #if ispointinside(userpoint, poly):
            

#     print (inside)