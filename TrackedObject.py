import trimesh
import fcl 
import numpy as np
 
class TrackedObject:
    def __init__(self, mesh, object_id, mode, node_name):
        self.trimesh = mesh
        self.matrix = np.eye(4)
        self.bvh = trimesh.collision.mesh_to_BVH(mesh)
        self.object_id = object_id
        self.channel = mode
        self.collision = False
        self.renderNode = None
        self.node_name = node_name
        self.has_slices = False
        self.slice_list = []

    def slice_self(self, x_dim=2, y_dim=2, z_dim=2):
            self.has_slices = True
            self.col_manager = fcl.DynamicAABBTreeCollisionManager()
            self.x_dim, self.y_dim, self.z_dim = x_dim, y_dim, z_dim

            # init all slices as tracked objects
            meshlist = self.slice3D(self.trimesh, x_dim, y_dim, self.z_dim)
            for count,this_mesh in enumerate(meshlist):
                if len(this_mesh[0].vertices) > 0: # Don't add the empty ones
                    newname = self.node_name + str(count)
                    self.slice_list.append(TrackedObject(this_mesh[0], self.object_id, self.channel, newname))
            
            self.setup_manager()
    
    def slice3D(self, currmesh, x_dim=2, y_dim=2, z_dim=2):
        print("Begin slice")
        obs = []
        #obj_dict = defaultdict(list)

        # Get bounds along desired axis
        boxbounds = currmesh.bounding_box.bounds
        #print(boxbounds)
        x_levels  = np.linspace((boxbounds[0][0]), (boxbounds[1][0]), num=x_dim+1, endpoint=True)
        y_levels  = np.linspace((boxbounds[0][1]), (boxbounds[1][1]), num=y_dim+1, endpoint=True)
        z_levels  = np.linspace((boxbounds[0][2]), (boxbounds[1][2]), num=z_dim+1, endpoint=True)

        for x, xval in enumerate(x_levels[:-1]):
            xpos = trimesh.intersections.slice_mesh_plane(currmesh, [1,0,0], [xval,0,0])
            xslice = trimesh.intersections.slice_mesh_plane(xpos, [-1,0,0], [x_levels[x+1],0,0])
            for y, yval in enumerate(y_levels[:-1]):
                ypos = trimesh.intersections.slice_mesh_plane(xslice, [0,1,0], [0,yval,0])
                yslice = trimesh.intersections.slice_mesh_plane(ypos, [0,-1,0], [0,y_levels[y+1],0])
                for z, zval in enumerate(z_levels[:-1]):
                    zpos = trimesh.intersections.slice_mesh_plane(yslice, [0,0,1], [0,0,zval])
                    zslice = trimesh.intersections.slice_mesh_plane(zpos, [0,0,-1], [0,0,z_levels[z+1]])
                    obs.append((zslice, x, y, z))

        return obs
    
    # setup collision manager
    def setup_manager(self):
        collision_geometries = []
        collision_objects = []
        for tracked_object in self.slice_list:
            transform1 = fcl.Transform(tracked_object.matrix[:3, :3], tracked_object.matrix[:3, 3])
            collision_geometries.append(tracked_object.bvh)
            collision_objects.append(fcl.CollisionObject(tracked_object.bvh, transform1))
        self.col_manager.registerObjects(collision_objects)
        self.col_manager.setup()

        self.geom_id_to_obj = {id(geom):obj for geom,obj in zip(collision_geometries, self.slice_list)}
        return

    # Takes in a collision geometry and returns the tracked object
    def lookupGeom(self, coll_geom):
        tracked_geom = [self.geom_id_to_obj[id(coll_geom)]]
        return tracked_geom[0]

    def update_child_poses(self):
        # Update the child tracked objects
        for slice in self.slice_list:
            slice.matrix = self.matrix

        #Update the objects in the collision manager
        newTransform = fcl.Transform(self.matrix[:3, :3], self.matrix[:3, 3])
        child_collision_objects = self.col_manager.getObjects()
        for col_obj in child_collision_objects:
            col_obj.setTransform(newTransform)

    
    def grid_from_dict(self, object_dict, x_start, x_stop, y_start, y_stop, z_start, z_stop):
        grid_objs = []
        for x in range(x_start, x_stop, (-1 if x_start>x_stop else 1)):
            for y in range(y_start, y_stop, (-1 if y_start>y_stop else 1)):
                for z in range(z_start, z_stop, (-1 if z_start>z_stop else 1)):
                    #print(x, y, z)
                    #grid_objs.append(object_dict[(x, y, z)][0].val)
                    grid_objs.append(object_dict[(x, y, z)][0].collision)
        return grid_objs

    def map_from_dict(self, object_dict, x_dim, y_dim, z_dim):

        top = self.grid_from_dict(object_dict, 0, x_dim, 0, 1, z_dim-1, -1)
        top = np.reshape(top, (z_dim, x_dim), 'F')

        left = self.grid_from_dict(object_dict, 0, 1, 0, y_dim, z_dim-1, -1)
        left = np.reshape(left, (y_dim, z_dim))

        front = self.grid_from_dict(object_dict, 0, x_dim, 0, y_dim, 0, 1)
        front = np.reshape(front, (y_dim, x_dim), 'F')

        right = self.grid_from_dict(object_dict, x_dim-1, x_dim, 0, y_dim, 0, z_dim)
        right = np.reshape(right, (y_dim, z_dim))

        back = self.grid_from_dict(object_dict, x_dim-1, -1, 0, y_dim, z_dim-1, z_dim)
        back = np.reshape(back, (y_dim, x_dim), 'F')

        bottom = self.grid_from_dict(object_dict, 0, x_dim, y_dim-1, y_dim, 0, z_dim)
        bottom = np.reshape(bottom, (z_dim, x_dim), 'F')

        empty = np.full((z_dim, z_dim), 2)
        empty_2 = np.full((z_dim, x_dim), 2)

        # Create the final shape out of the sub matrices
        res1 = np.concatenate((empty, top, empty, empty_2), axis=1)
        res2 = np.concatenate((left, front, right, back), axis=1)
        res3 = np.concatenate((empty, bottom, empty, empty_2), axis=1)
        res = np.concatenate((res1, res2, res3), axis=0)

        print(res)

        return res
            
    def print_collision_matrix(object_list):
        col_mat = []
        for sub_object in object_list:
            if sub_object.collision == True:
                col_mat.append(1)
            else:
                col_mat.append(0)
        print(col_mat)
        return
    