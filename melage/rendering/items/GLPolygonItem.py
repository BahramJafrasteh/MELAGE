from OpenGL.GL import *
from ..GLGraphicsItem import GLGraphicsItem
import numpy as np
from PyQt5 import QtGui

class GLPolygonItem(GLGraphicsItem):
    def __init__(self, size=None, antialias=True, glOptions='translucent'):
        GLGraphicsItem.__init__(self)
        self.antialias = antialias
        self.setGLOptions(glOptions)

        # FIX 1: Initialize the data structure here so paint() doesn't fail on start
        self.polys = {}

        self.colors = [
            [1, 0, 1], [0, 1, 1], [1, 0, 1], [1, 0, 0],
            [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0]
        ]

    def update_points(self, polys):
        self.polys = polys
        # FIX 2: Tell the GL View to redraw specifically this item
        self.update()

    def paint(self):
        if not self.polys:
            return

        self.setupGLState()

        # --- KEY FIX START ---
        # Disable lighting so OpenGL uses the specific color you set with glColor
        glDisable(GL_LIGHTING)
        # Disable textures in case the parent view had them enabled
        glDisable(GL_TEXTURE_2D)
        # --- KEY FIX END ---
        glDisable(GL_DEPTH_TEST) # Disable depth test to ensure lines are visible
        if self.antialias:
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # Ensure depth testing doesn't hide lines behind other transparent objects
        # Use GL_ALWAYS or disable depth test if you want them to always appear on top
        # glDisable(GL_DEPTH_TEST)

        glLineWidth(5.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        r = 0
        keys = list(self.polys.keys())  # Convert to list for safe indexing

        for key in keys:
            poly = self.polys[key]
            #print(poly)
            if len(poly) < 2:
                continue

            # Cycle colors safely
            color = self.colors[r % len(self.colors)]
            r += 1

            glBegin(GL_LINE_STRIP)
            # Set color BEFORE defining vertices
            glColor4f(color[0], color[1], color[2], 1.0)

            for p in poly:
                glVertex3f(p[0], p[1], p[2])

            glEnd()

        glEnable(GL_DEPTH_TEST) # Re-enable depth test if it was disabled