"""Examples of using pyrender for viewing and offscreen rendering.
"""
import pyglet
pyglet.options['shadow_window'] = False
import os
import numpy as np
import trimesh

from pyrender import PerspectiveCamera,IntrinsicsCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     Viewer, OffscreenRenderer, RenderFlags

#==============================================================================
# Mesh creation
#==============================================================================

#------------------------------------------------------------------------------
# Creating textured meshes from trimeshes
#------------------------------------------------------------------------------

# Drill trimesh
#spam_trimesh = trimesh.load('spam.obj')
spam_trimesh = trimesh.load("/media/bella/bellssd2/FoundationPose/org_tests/mesh/bell_v2_centered.obj", force='mesh')

pose1 = np.array([[9.992480874061584473e-01, -3.543521091341972351e-02, 1.573328673839569092e-02, 1.522932760417461395e-01],
                    [1.456845179200172424e-02, -3.289474546909332275e-02, -9.993526935577392578e-01, 5.868251994252204895e-02],
                    [3.592993691563606262e-02, 9.988304376602172852e-01, -3.235376998782157898e-02, 2.336114227771759033e-01],
                    [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])

def matrixToPose(transmatrix):
    # Transform eq taken from tracknet
    glcam_in_cvcam = np.array([[1,0,0,0],
                                [0,-1,0,0],
                                [0,0,-1,0],
                                [0,0,0,1]])
    cvcam_in_glcam = np.linalg.inv(glcam_in_cvcam)
    scene_pose = cvcam_in_glcam.dot(transmatrix)
    return scene_pose

pose_formed = matrixToPose(pose1)

def slice3D(currmesh, x_dim=2, y_dim=2, z_dim=2):
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

spam_meshes = slice3D(spam_trimesh)
print(spam_meshes)

#==============================================================================
# Light creation
#==============================================================================

direc_l = DirectionalLight(color=np.ones(3), intensity=1.0)
spot_l = SpotLight(color=np.ones(3), intensity=10.0,
                   innerConeAngle=np.pi/16, outerConeAngle=np.pi/6)
point_l = PointLight(color=np.ones(3), intensity=10.0)

#==============================================================================
# Camera creation
#==============================================================================

cam = PerspectiveCamera(yfov=(np.pi / 3.0))
cam_pose = np.array([
    [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 0.5],
    [1.0, 0.0,           0.0,           0.0],
    [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 0.4],
    [0.0,  0.0,           0.0,          1.0]
])
incam = IntrinsicsCamera(fx=608,fy=609,cx=320,cy=241,znear=0.1,zfar=100.0)

#==============================================================================
# Scene creation
#==============================================================================

scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))

#==============================================================================
# Adding objects to the scene
#==============================================================================
transparent_color = MetallicRoughnessMaterial(metallicFactor=0.5,alphaMode='BLEND',baseColorFactor=np.array([0.1, 1, 0.1, 0.3]))
opaque_color = MetallicRoughnessMaterial(metallicFactor=0.5,alphaMode='OPAQUE',baseColorFactor=np.array([0.5, 0.1, 0.1, 0.9]))

for spam_mesh in spam_meshes:
    print(spam_mesh[0])
    pyrender_mesh = Mesh.from_trimesh(spam_mesh[0], smooth=False, material = transparent_color, wireframe = False)
    name_of_node = ("x" + str(spam_mesh[1]) + "y" + str(spam_mesh[2]) + "z" + str(spam_mesh[3]))
    mesh_node = Node(name=name_of_node, matrix=pose_formed, mesh=pyrender_mesh)
    scene.add_node(mesh_node)

#==============================================================================
# Using the viewer with a default camera
#==============================================================================

v = Viewer(scene, shadows=True)

#==============================================================================
# Using the viewer with a pre-specified camera
#==============================================================================
#cam_node = scene.add(cam, pose=cam_pose)

camera_node = Node(camera=incam)
camera_node = scene.add_node(camera_node)

#v = Viewer(scene, central_node=drill_node)

#==============================================================================
# Rendering offscreen from that camera
#==============================================================================

r = OffscreenRenderer(viewport_width=640*2, viewport_height=480*2)

#plt.imshow(color)
#plt.show()

#==============================================================================
# Segmask rendering
#==============================================================================

# nm = {node: 20*(i + 1) for i, node in enumerate(scene.mesh_nodes)}
# seg = r.render(scene, RenderFlags.SEG, nm)[0]
# plt.figure()
# plt.imshow(seg)
# plt.show()

# r.delete()

#==============================================================================
# Render one-by-one?
#==============================================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(2, 4))
n = 0

for currnode in scene.mesh_nodes:
    # get mesh in the node
    rendermesh = currnode.mesh
    # get mesh primitives
    myprims = rendermesh.primitives[0]
    # Update material primitive
    # set mesh primitives 
    #myprims.material(newColor)
    myprims.material = opaque_color

    color, depth = r.render(scene)

    print("currnode", currnode.name)

    # Plot output
    ax = plt.subplot(2, 4, n + 1)
    ax.title.set_text(currnode.name)
    ax.imshow(color)
    ax.set_xticks([])
    ax.set_yticks([])
    n = n+1

    myprims.material = transparent_color


plt.show()

