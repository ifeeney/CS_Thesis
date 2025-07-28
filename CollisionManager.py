import fcl
from collections import defaultdict
import numpy as np

class CollisionManager:
    def __init__(self, channel):
        self.tracked_objects = []
        self.tracked_slices = []
        self.channel = channel

    def register_object(self, this_obj):
        if(self.channel == "All" or this_obj.channel == self.channel):
            self.tracked_objects.append(this_obj)
            if this_obj.has_slices is True:
                for slice in this_obj.slice_list:
                    self.tracked_slices.append(slice)

    def init_global_manager(self):
        self.global_manager = fcl.DynamicAABBTreeCollisionManager()
        self.global_manager.registerObjects(self.tracked_slices)
        self.global_manager.setup()

    def global_collision_callback(self, o1, o2, cdata):
        t1 = self.geom_id_to_tracked_obj.get(id(o1.collision_geometry))
        t2 = self.geom_id_to_tracked_obj.get(id(o2.collision_geometry))

        # Skip self-collisions
        if t1 is None or t2 is None:
            return False
        if t1.object_id == t2.object_id:
            return False

        # Record collision
        t1.collision = True
        t2.collision = True

        self.update_object_collision(t1, True)
        self.update_object_collision(t2, True)

        # Optionally store
        self.segment_collisions.append((t1, t2))

        return True  # Continue checking other pairs

    def channel_collision_check(self):

        # Group TrackedObjects by channel
        channel_groups = defaultdict(list)
        for obj in self.tracked_objects:  # Replace with your actual list
            channel_groups[obj.channel].append(obj)

        # Run collision checks only between objects within the same channel
        for channel, objs in channel_groups.items():

            if len(objs) != 2:
                print(f"⚠️ Channel '{channel}' has {len(objs)} objects; expected 2. Skipping.")
                continue

            obj1, obj2 = objs

            # If objects are whole, run one-to-one collision check

            # Otherwise run collision check using their own collision managers
            result = self.manager_to_manager_collision_check(obj1.col_manager, obj2.col_manager)

            print("Collision on channel " + channel + "= " + str(result.is_collision))

            if result.is_collision:
                for contact in result.contacts:

                    # Get the tracked objects
                    tracked_in_col1 = obj1.lookupGeom(contact.o1)
                    tracked_in_col2 = obj2.lookupGeom(contact.o2)

                    tracked_in_col1.collision = True
                    tracked_in_col2.collision = True

            else:
                pass

    def update_poses(self):
        for this_obj in self.tracked_objects:
            if this_obj.has_slices is True:
                this_obj.update_child_poses()

    def reset_collision(self):
        for obj in self.tracked_objects:
            obj.collision = False
        for slice in self.tracked_slices:
            slice.collision = False

    def update_object_collisions(self):
        for this_obj in self.tracked_objects:
            for compare_obj in self.tracked_objects:
                # Only check object with different IDs on the same channel
                if (this_obj.object_id != compare_obj.object_id) and (this_obj.channel == compare_obj.channel):
                    print("Checking collision on item {} with item {}".format(this_obj.object_id, compare_obj.object_id))
                    
                    if this_obj.has_slices is True and compare_obj.has_slices is True:
                        result_data = self.manager_to_manager_collision_check(this_obj.col_manager, compare_obj.col_manager)
                    else:
                        self.one_to_one_collision_check(this_obj, compare_obj)

    def check_object_collisions(self, mode="Manager"):
        for this_obj in self.tracked_objects:
            for compare_obj in self.tracked_objects:
                if this_obj.object_id != compare_obj.object_id:
                    print("Checking collision on item {} with item {}".format(this_obj.object_id, compare_obj.object_id))
                    if(mode=="Manager"):
                        result_data = self.manager_to_manager_collision_check(this_obj.col_manager, compare_obj.col_manager)
                        if result_data.is_collision:
                            objs_in_collision = set()
                            print("In collision with:", compare_obj.object_id)
                            for contact in result_data.contacts:
                  
                                # Get the tracked objects
                                tracked_in_col1 = this_obj.lookupGeom(contact.o1)
                                tracked_in_col2 = compare_obj.lookupGeom(contact.o2)

                                objs_in_collision.add(tracked_in_col1)
                                objs_in_collision.add(tracked_in_col2)

                            #result_dict[str(key)][str(object_id)] = "True"
                            for tracked_obj in objs_in_collision:
                                self.update_object_collision(tracked_obj, True)
                            break
                        else:
                            pass
                            #result_dict[str(key)][str(object_id)] = "False"
                    else:
                        if self.one_to_one_collision_check(this_obj, compare_obj):
                            print("In collision with:", compare_obj.object_id)
                            break

    # two single scene objects
    def one_to_one_collision_check(self, tracked_object1, tracked_object2):
        result = self.tracked_object_collision_check(tracked_object1, tracked_object2)
        if(result):
            tracked_object1.collision = True
            tracked_object2.collision = True
        return result

    def tracked_object_collision_check(self, tracked_object1, tracked_object2):

        transform1 = fcl.Transform(tracked_object1.matrix[:3, :3], tracked_object1.matrix[:3, 3])
        transform2 = fcl.Transform(tracked_object2.matrix[:3, :3], tracked_object2.matrix[:3, 3])

        collsision_obj_primary = fcl.CollisionObject(tracked_object1.bvh, transform1)
        collsision_obj_secondary = fcl.CollisionObject(tracked_object2.bvh, transform2)

        req = fcl.CollisionRequest(enable_contact=True)
        res = fcl.CollisionResult()

        n_contacts = fcl.collide(collsision_obj_primary, collsision_obj_secondary, req, res)
        
        #self.update_object_collision(tracked_object1, res.is_collision)
        #tracked_object2.update_collision(res.is_collision)

        return res.is_collision

    # two managers with many meshes
    def manager_to_manager_collision_check(self, manager1, manager2):
        collide_objs = manager1.getObjects()
        req = fcl.CollisionRequest(num_max_contacts=100, enable_contact=True)
        rdata = fcl.CollisionData(req, fcl.CollisionResult())
        manager1.collide(manager2, rdata, fcl.defaultCollisionCallback) #global_collision_callback
        #print ('Collision between manager 1 and manager 2?', rdata.result.is_collision)
        return rdata.result
    
    def update_object_collision(self, target_object, col_result):
        if (col_result != target_object.collision):
            target_object.collision = col_result
            self.viewer.update_object_color(target_object)
        target_object.collision = col_result
        return

    # helper to do transforms
    def mat_from_tf(self, translation, rotation):
        # Create a 4x4 transformation matrix
        #rotation_matrix = R.from_quat(rotation).as_matrix()
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation
        transform_matrix[:3, 3] = translation
        return transform_matrix

