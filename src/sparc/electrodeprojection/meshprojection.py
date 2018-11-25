"""
Created on 30 Oct, 2018 from mapclientplugins.meshgeneratorstep.

@author: Jesse Khorasanee
"""

import numpy as np

from opencmiss.zinc.glyph import Glyph
from opencmiss.zinc.node import Node
from opencmiss.zinc.field import Field


class MeshProjection(object):

    def __init__(self, region):
        self._region = region
        self._scene = self._region.getScene()
        # self._child_region = region.createChild('grid_plane')
        self._starting_number = 1000
        self._global_coordinates = []
        self._plane_norm = None
        self._xi_coordinates = {
            'element': [],
            'coordinates': [],
            'elementID': []
        }

    def generate_grid_points_4(self, points_list, number_on_side, plane_normal):
        # We generate our grid points by having 4 points that we assign weightings to
        # based on how far we are away from them.

        # INPUTS:
        #   pointsList : a list of four points defined in any 3D space.
        #   number_on_side : the number of nodes we have on each side of our array
        # (ie 8 nodes if we have a 64 point grid).
        #   plane_normal : the normal of the plane we are creating points in. EG [0,0,1] for the x,y plane.

        # OUTPUTS:
        #   coord_list : a list of the number_on_side^2 (ie 64) points in 3D space.
        #       eg: [[1 , 0, 0], [1, .06, 0], [1, .12, 0], ... [0, 1, 0]]

        # Extra notes:
        #   The points are selected in a counter clockwise fashion
        # (start anywhere and make a trapezoid counterclockwise)

        p1 = points_list[0]
        p2 = points_list[1]
        p3 = points_list[2]
        p4 = points_list[3]

        ns1 = number_on_side - 1

        plane_normal_offset = .55  # For offsetting the solver to solve from outside the mesh -> on it
        # ^ set this to 0 if you want points on the original plane

        grid_coord = []
        for i in range(number_on_side):
            for j in range(number_on_side):
                # Create our weightings (since we are setting points in a ccwise fashion our diagonal is w3
                w1 = ((ns1 - i) * (ns1 - j)) / (ns1 ** 2)   # top left point ( or first point )
                w2 = (j / ns1) * (ns1 - i) / ns1            # top right point ( or second point )
                w3 = i * j / (ns1 ** 2)                     # Diagonal point ( or third point )
                w4 = (i / ns1) * (ns1 - j) / ns1            # bottom left point ( or fourth point )

                # Use our weightings to find coordinates of our new point
                x = p4[0] * w4 + p3[0] * w3 + p2[0] * w2 + p1[0] * w1
                y = p4[1] * w4 + p3[1] * w3 + p2[1] * w2 + p1[1] * w1
                z = p4[2] * w4 + p3[2] * w3 + p2[2] * w2 + p1[2] * w1

                grid_coord.append([x, y, z])

        # offset our points if we want to
        plane_norm = np.array(plane_normal)
        self._plane_norm = plane_norm
        coord_list = []
        for i in range(len(grid_coord)):
            shifted_point = grid_coord[i] + plane_norm * plane_normal_offset
            coord_list.append(shifted_point.tolist())

        return coord_list

    def points_to_nodes(self, grid_list):

        fm = self._region.getFieldmodule()
        cache = fm.createFieldcache()
        coordinates = fm.findFieldByName('coordinates')
        coordinates = coordinates.castFiniteElement()
        nodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        node_template = nodes.createNodetemplate()
        node_template.defineField(coordinates)
        node_template.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_VALUE, 1)

        for i, grid_point in enumerate(grid_list):
            eeg_node = nodes.createNode(self._starting_number + i, node_template)
            cache.setNode(eeg_node)
            coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, grid_point)

    def render_nodes(self):
        self._scene.beginChange()
        node_points = self._scene.createGraphicsPoints()
        material_module = self._scene.getMaterialmodule()
        fm = self._region.getFieldmodule()
        node_points.setFieldDomainType(Field.DOMAIN_TYPE_NODES)
        coordinates = fm.findFieldByName('coordinates')
        coordinates = coordinates.castFiniteElement()
        node_points.setFieldDomainType(Field.DOMAIN_TYPE_NODES)
        node_points.setCoordinateField(coordinates)
        node_points.setMaterial(material_module.findMaterialByName('blue'))
        node_points.setVisibilityFlag(True)

        node_point_attr = node_points.getGraphicspointattributes()
        node_point_attr.setGlyphShapeType(Glyph.SHAPE_TYPE_SPHERE)
        node_point_attr.setBaseSize([.02, .02, .02])
        cmiss_number = fm.findFieldByName('cmiss_number')
        node_point_attr.setLabelField(cmiss_number)

        self._scene.endChange()

    def solve_nodes(self):
        fm = self._region.getFieldmodule()
        cache = fm.createFieldcache()
        self._global_coordinates = []
        self._xi_coordinates = {
            'element': [],
            'coordinates': [],
            'elementID': []
        }
        for i in range(0, 64):
            self.move_node(self._starting_number + i, plane_normal=self._plane_norm, cache=cache)

    def move_node(self, node_key, plane_normal=None, cache=None):
        # moveNode uses dot products combined with opencmiss' evaluateMeshLocation function
        # to solve where a point lies along a given normal (line in 3D).

        # usage: Please not that this solver assumes that the user knows where the solution should be in one dimension,
        # for example: Where is the closest mesh point at x=0 (normal=[1,0,0]) starting at [1,3,2]?

        # Inputs:
        #   region: the region you wish to solve in (must contain a mesh and a the node you wish to move)
        #   node_key: identifier of the node we wish to project onto the mesh
        #   plane_normal: the direction we wish to project the node onto the mesh
        #   cache: (optional) pass the cache to this function to enhance performance

        # Outputs:
        #   new_coordinates: the new coordinates of the node now that it is on the mesh
        # (or if we could not solve it will be the last iteration)

        # Adjust the solving parameters here:
        max_iterations = 20
        tol = .001
        plane_normal_offset = 0

        if plane_normal is None:
            plane_normal = [0, 1, 0]

        # Re-acquire opencmiss.zinc variables
        fm = self._region.getFieldmodule()
        coordinates = fm.findFieldByName('coordinates')
        coordinates = coordinates.castFiniteElement()
        if cache is None:
            cache = fm.createFieldcache()

        # Create templates
        nodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        node_template = nodes.createNodetemplate()
        node_template.defineField(coordinates)
        node_template.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_VALUE, 1)

        # Create our first new node for the search
        old_node = nodes.findNodeByIdentifier(node_key)
        cache.setNode(old_node)
        [_, old_coordinates] = coordinates.evaluateReal(cache, 3)

        # Create our mesh search
        mesh = fm.findMeshByName('mesh3d')
        found_mesh_location = fm.createFieldFindMeshLocation(coordinates, coordinates, mesh)
        found_mesh_location.setSearchMode(found_mesh_location.SEARCH_MODE_NEAREST)

        # shift the point in preparation for first iteration
        plane_norm = np.array(plane_normal)
        shifted_point = old_coordinates + plane_norm * plane_normal_offset
        coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, shifted_point.tolist())

        # initialise for our solver
        it = 1
        old_coordinates = shifted_point
        test_coordinates = [10, 10, 10]
        new_coordinates = shifted_point
        el = None

        while abs(np.linalg.norm(np.dot(test_coordinates, plane_norm) - np.dot(new_coordinates, plane_norm))) > tol:
            # ^^ test if x and y changes are within tolerance by testing the magnitude of the
            # difference of the dot products in the direction of our normal

            # Find nearest mesh location using the evaluateMeshLocation function
            [el, coordinates] = found_mesh_location.evaluateMeshLocation(cache, 3)
            cache.setMeshLocation(el, coordinates)
            [_, mesh_coordinates] = coordinates.evaluateReal(cache, 3)

            # Update our search location by finding how far we have moved in the plane_normal direction
            new_coordinates = old_coordinates + np.dot(mesh_coordinates - old_coordinates, plane_norm) * plane_norm
            cache.setNode(old_node)
            coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, new_coordinates.tolist())

            # switch our test coordinates
            test_coordinates = old_coordinates
            old_coordinates = new_coordinates

            # Break in case we can not converge
            it += 1
            if it > max_iterations:
                print(f'Could not converge on node {node_key}')
                break

        print(f'Node {node_key} was solved in {it-1} iterations')
        self._xi_coordinates['element'].append(el)
        self._xi_coordinates['elementID'].append(el.getIdentifier())
        self._xi_coordinates['coordinates'].append(coordinates)
        self._global_coordinates.append(new_coordinates)


MeshProjection.moveNode = MeshProjection.move_node
MeshProjection.generateGridPoints4 = MeshProjection.generate_grid_points_4
MeshProjection.pointsToNodes = MeshProjection.points_to_nodes
MeshProjection.renderNodes = MeshProjection.render_nodes
MeshProjection.solveNodes = MeshProjection.solve_nodes
