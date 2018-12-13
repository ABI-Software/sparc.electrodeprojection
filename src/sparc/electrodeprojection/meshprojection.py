"""
Created on 30 Oct, 2018 from mapclientplugins.meshgeneratorstep.

@author: Jesse Khorasanee
"""

import numpy as np

from opencmiss.zinc.glyph import Glyph
from opencmiss.zinc.node import Node
from opencmiss.zinc.field import Field, FieldFindMeshLocation
from opencmiss.zinc.element import Element


class MeshProjection(object):

    def __init__(self, region):
        self._region = region
        self._width = -1
        self._height = -1
        self._depth = 500
        self._z_scale = 0.5
        self._offset_field = None
        self._projection_matrix_field = None
        self._scaled_image_coordinates = None

        self._scene = self._region.getScene()
        # self._child_region = region.createChild('grid_plane')
        self._starting_number = 1000
        self._time = 0.0
        self._global_coordinates = []
        self._plane_norm = np.array([0, 0, 1])
        self._xi_coordinates = {
            'element': [],
            'coordinates': [],
            'elementID': []
        }

    def set_time(self, time):
        self._time = time

    def get_scene(self):
        return self._scene

    def render(self):

        if self._scaled_image_coordinates is None:
            self._initialise()

        self._scene.beginChange()
        material_module = self._scene.getMaterialmodule()
        field_module = self._region.getFieldmodule()
        field_module.defineAllFaces()

        coordinates = field_module.findFieldByName('coordinates')
        blue_material = material_module.findMaterialByName('blue')
        green_material = material_module.findMaterialByName('green')
        yellow_material = material_module.findMaterialByName('yellow')
        cyan_material = material_module.findMaterialByName('cyan')

        axes = self._scene.createGraphicsPoints()
        axes_point_attr = axes.getGraphicspointattributes()
        axes_point_attr.setGlyphShapeType(Glyph.SHAPE_TYPE_AXES_XYZ)
        axes_point_attr.setBaseSize([self._width, self._height, self._depth])
        axes.setMaterial(cyan_material)

        lines = self._scene.createGraphicsLines()
        lines.setCoordinateField(self._scaled_image_coordinates)
        lines.setMaterial(blue_material)
        lines.setExterior(True)
        # lines_attributes

        lines = self._scene.createGraphicsLines()
        lines.setCoordinateField(coordinates)
        lines.setMaterial(green_material)
        lines.setExterior(True)

        node_points = self._scene.createGraphicsPoints()
        node_points.setFieldDomainType(Field.DOMAIN_TYPE_MESH3D)
        node_points.setCoordinateField(coordinates)
        node_points.setMaterial(green_material)
        # node_points.setVisibilityFlag(True)

        node_point_attr = node_points.getGraphicspointattributes()
        node_point_attr.setGlyphShapeType(Glyph.SHAPE_TYPE_SPHERE)
        node_point_attr.setBaseSize([.02, .02, .02])

        node_points = self._scene.createGraphicsPoints()
        node_points.setFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        node_points.setCoordinateField(coordinates)
        node_points.setMaterial(yellow_material)
        node_point_attr = node_points.getGraphicspointattributes()
        node_point_attr.setGlyphShapeType(Glyph.SHAPE_TYPE_SPHERE)
        node_point_attr.setBaseSize([8, 8, 8])

        self._scene.endChange()

    def set_width_and_height(self, width, height):
        self._width = width
        self._height = height

    def set_depth(self, depth):
        """
        Set the depth to represent the eye location **after** setting the width and height.

        :param depth:  The depth of the image plane away from the eye.
        :return:
        """
        self._depth = depth
        if self._scaled_image_coordinates is None:
            self._initialise()
        else:
            self._update_depth()

    def get_depth(self):
        return self._depth

    def get_z_scale_factor(self):
        return self._z_scale

    def set_z_scale_factor(self, scale_factor):
        self._z_scale = scale_factor

    def project_point(self, point):
        if self._depth == -1 or self._width == -1 or self._height == -1:
            raise Exception('Required dimensions are not set, currently [{0}, {1}, {2}]'
                            .format(self._width, self._height, self._depth))
        if self._scaled_image_coordinates is None:
            raise Exception('Scaled image coordinates are not set but the image dimensions are????')

        field_module = self._region.getFieldmodule()
        mesh = field_module.findMeshByDimension(2)

        field_module.beginChange()
        mesh_group_field = field_module.createFieldElementGroup(mesh)
        is_exterior_field = field_module.createFieldIsExterior()
        xi_3_1_field = field_module.createFieldIsOnFace(Element.FACE_TYPE_XI3_1)
        and_field = field_module.createFieldAnd(is_exterior_field, xi_3_1_field)
        mesh_group = mesh_group_field.getMeshGroup()
        mesh_group.addElementsConditional(and_field)

        point_field = field_module.createFieldConstant([point[0], point[1], self._z_scale * self._depth])
        field_find_mesh_location = field_module.createFieldFindMeshLocation(point_field,
                                                                            self._scaled_image_coordinates,
                                                                            mesh_group)
        field_find_mesh_location.setSearchMode(FieldFindMeshLocation.SEARCH_MODE_NEAREST)

        field_cache = field_module.createFieldcache()
        field_cache.setTime(self._time)

        found_element, xi_location = field_find_mesh_location.evaluateMeshLocation(field_cache, 2)
        location = self._get_coordinates(found_element, xi_location)
        self._create_data_point(location)

        field_module.endChange()

        return [found_element, xi_location]

    def clear_projected_points(self):
        field_module = self._region.getFieldmodule()
        node_set = field_module.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        node_set.destroyAllNodes()

    def _get_coordinates(self, element, xi_location):
        field_module = self._region.getFieldmodule()
        field_cache = field_module.createFieldcache()
        field_cache.setMeshLocation(element, xi_location)
        coordinates = field_module.findFieldByName('coordinates')
        result, location = coordinates.evaluateReal(field_cache, 3)

        return location

    def _create_data_point(self, location):
        field_module = self._region.getFieldmodule()
        coordinates = field_module.findFieldByName('coordinates')
        node_set = field_module.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        node_template = node_set.createNodetemplate()
        node_template.defineField(coordinates)
        node = node_set.createNode(-1, node_template)
        field_cache = field_module.createFieldcache()
        field_cache.setNode(node)
        coordinates.assignReal(field_cache, location)

    def _calculate_offset(self):
        return [-(self._width + 0.5) / 2, -(self._height + 0.5) / 2, 0.0]

    def _calculate_projection_matrix(self):
        return [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, -1.0 / self._depth, 1.0]

    def _update_depth(self):
        field_module = self._region.getFieldmodule()
        field_cache = field_module.createFieldcache()
        offset = self._calculate_offset()
        projection_matrix = self._calculate_projection_matrix()
        field_module.beginChange()
        self._offset_field.assignReal(field_cache, offset)
        self._projection_matrix_field.assignReal(field_cache, projection_matrix)
        field_module.endChange()

    def _initialise(self):
        self._initialised = True
        field_module = self._region.getFieldmodule()
        field_module.beginChange()

        coordinates_field = field_module.findFieldByName('coordinates')
        self._offset_field = field_module.createFieldConstant(self._calculate_offset())
        offset_coordinate_field = field_module.createFieldAdd(coordinates_field, self._offset_field)
        self._projection_matrix_field = field_module.createFieldConstant(self._calculate_projection_matrix())
        projected_coordinates = field_module.createFieldProjection(offset_coordinate_field,
                                                                   self._projection_matrix_field)
        image_coordinates = field_module.createFieldSubtract(projected_coordinates, self._offset_field)
        flattening_field = field_module.createFieldConstant([1.0, 1.0, 1.0])
        self._scaled_image_coordinates = field_module.createFieldMultiply(image_coordinates, flattening_field)
        self._scaled_image_coordinates = coordinates_field
        field_module.endChange()

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

        plane_normal_offset = .15  # For offsetting the solver to solve from outside the mesh -> on it
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

    def find_location(self, x, y):
        pass

    def solve_nodes(self):
        fm = self._region.getFieldmodule()
        cache = fm.createFieldcache()
        self._global_coordinates = []
        self._xi_coordinates = {
            'element': [],
            'coordinates': [],
            'elementID': []
        }
        data_points_node_set = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        node_set_iterator = data_points_node_set.createNodeiterator()
        node = node_set_iterator.next()
        count = 1
        while node.isValid() and count == 1:
            self.move_node(node, plane_normal=self._plane_norm, cache=cache)
            node = node_set_iterator.next()
            count += 1
        # for i in range(0, 64):
        #     self.move_node(self._starting_number + i, plane_normal=self._plane_norm, cache=cache)

    def move_node(self, data_point, plane_normal=None, cache=None):
        # moveNode uses dot products combined with opencmiss' evaluateMeshLocation function
        # to solve where a point lies along a given normal (line in 3D).

        # usage: Please note that this solver assumes that the user knows where the solution should be in one dimension,
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
        plane_normal_offset = .55

        if plane_normal is None:
            plane_normal = [0, 1, 0]

        # Re-acquire opencmiss.zinc variables
        fm = self._region.getFieldmodule()
        coordinates_field = fm.findFieldByName('coordinates')
        coordinates_field = coordinates_field.castFiniteElement()
        if cache is None:
            cache = fm.createFieldcache()

        # Create templates
        # data_points = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        # node_template = data_points.createNodetemplate()
        # node_template.defineField(coordinates_field)
        # node_template.setValueNumberOfVersions(coordinates_field, -1, Node.VALUE_LABEL_VALUE, 1)

        # Create our first new node for the search
        # old_node = data_points.findNodeByIdentifier(node_key)

        cache.setTime(self._time)
        cache.setNode(data_point)
        [_, old_coordinates] = coordinates_field.evaluateReal(cache, 3)

        # Create our mesh search
        mesh = fm.findMeshByName('mesh3d')
        found_mesh_location = fm.createFieldFindMeshLocation(coordinates_field, coordinates_field, mesh)
        found_mesh_location.setSearchMode(found_mesh_location.SEARCH_MODE_NEAREST)

        # shift the point in preparation for first iteration
        plane_norm = np.array(plane_normal)
        shifted_point = old_coordinates + plane_norm * plane_normal_offset
        coordinates_field.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, shifted_point.tolist())

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
            [el, xi_coordinates] = found_mesh_location.evaluateMeshLocation(cache, 3)
            cache.setMeshLocation(el, xi_coordinates)
            [_, mesh_coordinates] = coordinates_field.evaluateReal(cache, 3)

            # Update our search location by finding how far we have moved in the plane_normal direction
            new_coordinates = old_coordinates + np.dot(mesh_coordinates - old_coordinates, plane_norm) * plane_norm
            # cache.setNode(data_point)
            coordinates_field.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, new_coordinates.tolist())

            # switch our test coordinates
            test_coordinates = old_coordinates
            old_coordinates = new_coordinates

            # Break in case we can not converge
            it += 1
            if it > max_iterations:
                print('Could not converge on node {0}'.format(data_point.getIdentifier()))
                break

        # print(f'Node {node_key} was solved in {it-1} iterations')
        self._xi_coordinates['element'].append(el)
        self._xi_coordinates['elementID'].append(el.getIdentifier())
        self._xi_coordinates['coordinates'].append(coordinates_field)
        self._global_coordinates.append(new_coordinates)

    def get_global_coordinates(self):
        return self._global_coordinates


MeshProjection.moveNode = MeshProjection.move_node
MeshProjection.generateGridPoints4 = MeshProjection.generate_grid_points_4
MeshProjection.pointsToNodes = MeshProjection.points_to_nodes
MeshProjection.renderNodes = MeshProjection.render_nodes
MeshProjection.solveNodes = MeshProjection.solve_nodes
