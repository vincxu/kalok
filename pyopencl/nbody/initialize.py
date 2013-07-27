from OpenGL.GL import *


import timing
timings = timing.Timing()

@timings
def fountain_np(num):
    """numpy way of initializing data using ufuncs instead of loops"""
    import numpy
    pos = numpy.ndarray((num, 4), dtype=numpy.float32)
    col = numpy.ndarray((num, 4), dtype=numpy.float32)
    vel = numpy.ndarray((num, 4), dtype=numpy.float32)

    pos[:,0] = numpy.sin(numpy.arange(0., num) * 2.001 * numpy.pi / num) 
    pos[:,0] *= numpy.random.random_sample((num,)) / 3. + .2
    pos[:,1] = numpy.cos(numpy.arange(0., num) * 2.001 * numpy.pi / num) 
    pos[:,1] *= numpy.random.random_sample((num,)) / 3. + .2
    pos[:,2] = 0.
    pos[:,3] = 1.

    col[:,0] = 0.
    col[:,1] = 1.
    col[:,2] = 0.
    col[:,3] = 1.

    vel[:,0] = pos[:,0] * 2.
    vel[:,1] = pos[:,1] * 2.
    vel[:,2] = 3.
    vel[:,3] = numpy.random.random_sample((num, ))

    return pos, col, vel
    
@timings
def fountain_loopy(num):
    """This is a slower way of initializing the points (by 10x for large num)
    but more illustrative of whats going on""" 
    
    from math import sqrt, sin, cos
    import numpy
    pos = numpy.ndarray((num, 4), dtype=numpy.float32)
    col = numpy.ndarray((num, 4), dtype=numpy.float32)
    vel = numpy.ndarray((num, 4), dtype=numpy.float32)

    import random
    random.seed()
    for i in xrange(0, num):
        radius = random.uniform(1.0, 2.0)
        theta = random.uniform(0.0, 2*3.14);
        phi = random.uniform(-3.14, 3.14);
        ctemp = radius*cos(phi)
        x = ctemp*cos(theta)
        y = ctemp*sin(theta) 
        z = radius*sin(phi)

        pos[i,0] = x 
        pos[i,1] = y 
        pos[i,2] = z 
        pos[i,3] = 1.0

        rand_col = random.uniform(0.,1.0)
        col[i,0] = 1.0
        rand_col = random.uniform(0.,1.0)
        col[i,1] = 0.0
        rand_col = random.uniform(0.,1.0)
        col[i,2] = 0.0 
        col[i,3] = 1.

        life = random.uniform(0.0,3.0)
        veloc_rand = 1.0
        norme = sqrt(z*y*y*z + x*x*z*z + 4*x*x*y*y)
        vel[i,0] = z*y * veloc_rand / norme
        vel[i,1] = x*y * veloc_rand / norme
        vel[i,2] = -2 * x * y * veloc_rand / norme
        vel[i,3] = 0.0 

    return pos, col, vel


def fountain(num):
    """Initialize position, color and velocity arrays we also make Vertex
    Buffer Objects for the position and color arrays"""

    #pos, col, vel = fountain_np(num)
    pos, col, vel = fountain_loopy(num)
    
    print timings

    #create the Vertex Buffer Objects
    from OpenGL.arrays import vbo 
    pos_vbo = vbo.VBO(data=pos, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
    pos_vbo.bind()
    col_vbo = vbo.VBO(data=col, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
    col_vbo.bind()

    return (pos_vbo, col_vbo, vel)


