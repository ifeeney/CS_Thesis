import pyrender
import numpy as np

class ViewerUtils:
    def toggle_mod_nodes(viewer):
        viewer.mod_visible = not getattr(viewer, 'mod_visible', True)

        for node in getattr(viewer, 'mod_nodes', []):
            try:
                if viewer.mod_visible:
                    viewer.scene.add_node(node)
                    viewer._message_text = "MOD NODES ON"
                else:
                    viewer.scene.remove_node(node)
                    viewer._message_text = "MOD NODES OFF"
            except Exception:
                pass
        print(f"[viewer] mod_nodes {'visible' if viewer.mod_visible else 'hidden'}")

    def toggle_fp_nodes(viewer):
        viewer.fp_visible = not getattr(viewer, 'fp_visible', True)

        for node in getattr(viewer, 'fp_nodes', []):
            try:
                if viewer.fp_visible:
                    viewer.scene.add_node(node)
                    viewer._message_text = "FP NODES ON"
                else:
                    viewer.scene.remove_node(node)
                    viewer._message_text = "FP NODES OFF"
            except Exception:
                pass
        print(f"[viewer] fp_nodes {'visible' if viewer.fp_visible else 'hidden'}")

    def toggle_gt_nodes(viewer):
        viewer.gt_visible = not getattr(viewer, 'gt_visible', True)

        for node in getattr(viewer, 'gt_nodes', []):
            try:
                if viewer.gt_visible:
                    viewer.scene.add_node(node)
                    viewer._message_text = "GT NODES ON"
                else:
                    viewer.scene.remove_node(node)
                    viewer._message_text = "GT NODES OFF"
            except Exception:
                pass
        print(f"[viewer] gt_nodes {'visible' if viewer.gt_visible else 'hidden'}")

class CollisionViewer:
    def __init__(self, channel, cam="Bell"):
        # Initialize scene
        self.scene = pyrender.Scene(ambient_light = [0.3,0.3,0.3, 1.0])
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0)

        if cam == ("Bell"):
            incam = pyrender.IntrinsicsCamera(fx=608,fy=609,cx=320,cy=241,znear=0.1,zfar=100.0)
        elif cam == ("YCB"):
            incam = pyrender.IntrinsicsCamera(fx=1066.778,fy=1067.487,cx=312.9869,cy=241.3109,znear=0.1,zfar=1500.0)

        # Add the camera
        camera_node = pyrender.Node(camera=incam)
        self.camera_node = self.scene.add_node(camera_node)

        # Don't view until we publish
        self.viewer = None

        self.tracked_obj_list = [] # Nodes for rendering in pyrender scene
        self.bvh_list = [] # the same meshes as BVH for running collision detection on
        self.mod_nodes = []
        self.fp_nodes = []
        self.gt_nodes = []

        self.channel = channel

    def publish_scene(self, run_as_live=False):
        # If you want to update the viewer while it's running this must be set to True
        if(run_as_live):
            self.viewer = pyrender.Viewer(self.scene,
                    run_in_thread=True,
                    registered_keys=({"j": ViewerUtils.toggle_mod_nodes, 
                                      "k": ViewerUtils.toggle_fp_nodes, 
                                      "l": ViewerUtils.toggle_gt_nodes}),
                    refresh_rate = 200.0,  # Time for caption to disappear
                    use_raymond_lighting=True,
                    show_world_axis=False,
                    show_mesh_axes=True)
                        
            self.viewer.mod_nodes = self.mod_nodes
            self.viewer.mod_visible = True
            self.viewer.fp_nodes = self.fp_nodes
            self.viewer.fp_visible = True
            self.viewer.gt_nodes = self.gt_nodes
            self.viewer.gt_visible = True
        else:
            self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True,
                        show_world_axis=False,
                        show_mesh_axes=True)
            
    def view_offscreen(self):
        self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True,
                        show_world_axis=False,
                        show_mesh_axes=True)
        self.renderer = pyrender.OffscreenRenderer(viewport_width=640*2, viewport_height=480*2)
        color, depth = self.renderer(self.scene)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(color)
        plt.show()

    def update_object_pose(self, tracked_object):
        if tracked_object.renderNode not in self.scene.get_nodes():
            #print(f"Warning: Node {tracked_object.node_name} not found in the scene.")
            return

        new_pose = self.matrixToPose(tracked_object.matrix)
        self.viewer.render_lock.acquire()
        self.scene.set_pose(tracked_object.renderNode, new_pose)
        self.viewer.render_lock.release()

    def update_objects_in_scene(self):
        for obj in self.tracked_obj_list:
            self.update_object_pose(obj)

    def makeColor(self, type="Random", opacity=0.5):
        # Color is given in RGBA format
        if type=="None":
            return
        elif type=="Red":
            color = np.array([1, 0.1, 0.1, 0.8])
            mat = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.5,
                alphaMode='BLEND',
                baseColorFactor=color)
        elif type=="Green":
            color = np.array([0.1, 1, 0.1, 0.3])
            mat = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.5,
                alphaMode='BLEND',
                baseColorFactor=color)
        elif type=="Blue":
            color = np.array([0.1, 0.1, 1, 0.8])
            mat = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.5,
                alphaMode='BLEND',
                baseColorFactor=color)
        else:
            np.random.seed(None)  
            color = np.random.rand(4)
            color[3] = 1
            mat = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.5,
                alphaMode='OPAQUE',
                baseColorFactor=color
        ) 
        return mat 
    
    def update_object_color(self, tracked_object):
        # get mesh in the node
        rendermesh = tracked_object.renderNode.mesh
        # get mesh primitives
        myprims = rendermesh.primitives[0]
        # Update material primitive
        if(tracked_object.collision):
            newColor = self.makeColor("Red")
        else:
            newColor = self.makeColor("Green")
        # set mesh primitives 
        #print(myprims)
        myprims.material = newColor
        return

    def update_all_colors(self):
        for obj in self.tracked_obj_list:
            self.update_object_color(obj)
    
    def matrixToPose(self, transmatrix):
        # Transform eq taken from tracknet
        glcam_in_cvcam = np.array([[1,0,0,0],
                                    [0,-1,0,0],
                                    [0,0,-1,0],
                                    [0,0,0,1]])
        cvcam_in_glcam = np.linalg.inv(glcam_in_cvcam)
        scene_pose = cvcam_in_glcam.dot(transmatrix)
        return scene_pose

    def add_object_to_scene(self, tracked_object, color="Random"):

        scene_pose = self.matrixToPose(tracked_object.matrix)

        # Only add objects if they are on the chosen channel
        if(self.channel == "All" or tracked_object.channel == self.channel):

            if(tracked_object.channel == "Mod"): color = "Blue"
            elif(tracked_object.channel == "FP"): color = "Red"
            elif(tracked_object.channel == "GT"): color = "Green"

            if tracked_object.has_slices is True:
                for slice in tracked_object.slice_list:
                    pyrender_mesh = pyrender.Mesh.from_trimesh(slice.trimesh, smooth=False, material = self.makeColor("Random"), wireframe = False)
                    mesh_node = pyrender.Node(name=slice.node_name, matrix=scene_pose, mesh=pyrender_mesh)
                    self.scene.add_node(mesh_node)
                    slice.renderNode = mesh_node

                    if(slice.channel == "Mod"):
                        self.mod_nodes.append(mesh_node)
                    elif(slice.channel == "FP"):
                        self.fp_nodes.append(mesh_node)
                    else:
                        self.gt_nodes.append(mesh_node)
                    
                    self.tracked_obj_list.append(slice)

            else:
                pyrender_mesh = pyrender.Mesh.from_trimesh(tracked_object.trimesh, smooth=False, material = self.makeColor(color), wireframe = False)
                mesh_node = pyrender.Node(name=tracked_object.node_name, matrix=scene_pose, mesh=pyrender_mesh)
                self.scene.add_node(mesh_node)
                tracked_object.renderNode = mesh_node

                if(tracked_object.channel == "Mod"):
                    self.mod_nodes.append(mesh_node)
                elif(tracked_object.channel == "FP"):
                    self.fp_nodes.append(mesh_node)
                else:
                    self.gt_nodes.append(mesh_node)

                self.tracked_obj_list.append(tracked_object)

        return 
    
