import json
import os
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import fcl
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from EvaluatePoseUtils import *
from CollisionViewer import *
from TrackedObject import *
import time

def get_transform_params(obj):
    
    tx = obj.matrix[0, 3]
    ty = obj.matrix[1, 3]
    tz = obj.matrix[2, 3]

    # Extract rotation matrix (3x3)
    rotation_matrix = obj.matrix[0:3, 0:3]
    rotation = R.from_matrix(rotation_matrix)

    # Convert rotation matrix to Euler angles (in radians)
    #rx, ry, rz = rotation.as_euler('xyz', degrees=False)  
    #transform_params = [tx, ty, tz, rx, ry, rz]
    
    # Convert rotation matrix to rotation vector
    rotvec = rotation.as_rotvec()
    transform_params = [tx, ty, tz] + rotvec.tolist()
    
    return transform_params, obj.matrix
    
def tracked_obj_dist_check(tracked_object1, tracked_object2, mat1 = None, mat2 = None):
    if mat1 is not None: 
        obj1_matrix = mat1
    else:
        obj1_matrix = tracked_object1.matrix

    if mat2 is not None:
        obj2_matrix = mat2
    else:
        obj2_matrix = tracked_object2.matrix

    transform1 = fcl.Transform(obj1_matrix[:3, :3], obj1_matrix[:3, 3])
    transform2 = fcl.Transform(obj2_matrix[:3, :3], obj2_matrix[:3, 3])

    collsision_obj_primary = fcl.CollisionObject(tracked_object1.bvh, transform1)
    collsision_obj_secondary = fcl.CollisionObject(tracked_object2.bvh, transform2)

    request = fcl.DistanceRequest(enable_signed_distance = True)
    result = fcl.DistanceResult()

    ret = fcl.distance(collsision_obj_primary, collsision_obj_secondary, request, result)

    return ret*10
    
def tracked_obj_collision_check(tracked_object1, tracked_object2, mat1 = None, mat2 = None):
    if mat1 is not None: 
        obj1_matrix = mat1
    else:
        obj1_matrix = tracked_object1.matrix

    if mat2 is not None:
        obj2_matrix = mat2
    else:
        obj2_matrix = tracked_object2.matrix

    transform1 = fcl.Transform(obj1_matrix[:3, :3], obj1_matrix[:3, 3])
    transform2 = fcl.Transform(obj2_matrix[:3, :3], obj2_matrix[:3, 3])

    collsision_obj_primary = fcl.CollisionObject(tracked_object1.bvh, transform1)
    collsision_obj_secondary = fcl.CollisionObject(tracked_object2.bvh, transform2)

    req = fcl.CollisionRequest(num_max_contacts=500, enable_contact=True)
    res = fcl.CollisionResult()

    n_contacts = fcl.collide(collsision_obj_primary, collsision_obj_secondary, req, res)
    
    #tracked_object2.update_collision(res.is_collision)
    max_depth = 0
    max_norm = None

    for contact in res.contacts:
        if contact.penetration_depth >= max_depth:
            max_depth = contact.penetration_depth
            max_norm = contact.normal

    return res.is_collision, max_depth, max_norm

def equality_constraint(transform_params, tracked_obj1, tracked_obj2):

    tx, ty, tz, rx, ry, rz = transform_params

    # Create transformation matrix from tx, ty, tz and rotation angles rx, ry, rz
    #rotation = R.from_quat([q1,q2,q3,q4])
    #rotation = R.from_euler('xyz', [rx, ry, rz], degrees=False)
    rotation = R.from_rotvec([rx, ry, rz])
    rot_mat = R.as_matrix(rotation)
    transform_matrix = np.eye(4)  # Start with an identity matrix
    transform_matrix[:3, :3] = rot_mat  # Set the rotation part
    transform_matrix[:3, 3] = [tx, ty, tz]  # Set the translation part

    # Apply get distances
    # Try (pos_dist*10)**2 - neg_dist**2
    pos_dist = tracked_obj_dist_check(tracked_obj1, tracked_obj2, mat1=transform_matrix)
    _, neg_dist, norm = tracked_obj_collision_check(tracked_obj1, tracked_obj2, mat1=transform_matrix)
    distance = (pos_dist - neg_dist)
    #penalty = smooth_sigmoid_penalty(pos_dist) + smooth_sigmoid_penalty(neg_dist)
    
    return distance*10 # 10

def loss_from_init(transform_params, tracked_obj1, tracked_obj2):
    
    tx, ty, tz, rx, ry, rz = transform_params
    
    # penalty for large translations from tracked position
    diff_tx = abs(tracked_obj1.matrix[0, 3] - tx)
    diff_ty = abs(tracked_obj1.matrix[1, 3] - ty)
    diff_tz = abs(tracked_obj1.matrix[2, 3] - tz)
    translation_penalty = (30*diff_tx**2 + 30*diff_ty**2 + 200*diff_tz**2) #30 , 30, 200
    trans_unweighted = (diff_tx**2 + diff_ty**2 + diff_tz**2)
    
    # penalty for large rotations
    # target_rotation = R.from_matrix(tracked_obj1.matrix[:3, :3])
    # goalx, goaly, goalz = target_rotation.as_euler('xyz', degrees=False)
    # diff_rx = np.arctan2(np.sin(rx-goalx), np.cos(rx-goalx))
    # diff_ry = np.arctan2(np.sin(ry-goaly), np.cos(ry-goaly))
    # diff_rz = np.arctan2(np.sin(rz-goalz), np.cos(rz-goalz))
    # rotation_penalty =  (200*diff_rx**2 + 20*diff_ry**2 + 2*diff_rz**2) #8, 8, 2 #3, 3, 3
    # rot_unweighted = (diff_rx**2 + diff_ry**2 + diff_rz**2)

    # Relative rotation between target and estimate
    R_est = R.from_rotvec([rx, ry, rz]).as_matrix()
    R_target = tracked_obj1.matrix[:3, :3]
    R_rel = R_target.T @ R_est
    rotvec = R.from_matrix(R_rel).as_rotvec()  # axis-angle: axis * angle

    # Apply axis-specific weights
    weights = np.array([100.0, 100.0, 50.0])
    rotation_penalty = np.sum((rotvec * weights)**2)
    
    return (translation_penalty + rotation_penalty)

def read_fp_results(video_dir):
    object_poses = []
    hand_poses = []

    object_file_path = os.path.join(video_dir, "fp_results_cube_o.json")
    with open(object_file_path) as file:
        object_file = json.load(file)
    for frame,pose in object_file.items():
        object_poses.append(pose['Pose'])

    hand_file_path = os.path.join(video_dir, "fp_results_finger.json")
    with open(hand_file_path) as file:
        hand_file = json.load(file)
    for frame,pose in hand_file.items():
        hand_poses.append(pose['Pose'])

    return object_poses, hand_poses

def read_aruco_results(shapename, gt_setting, video_dir):

    aruco_id_finger = 6
    if(shapename == "bighex"):
        aruco_id = 0
    elif(shapename == "smallhex"):
        aruco_id = 11

    # Open arUco results
    if(gt_setting == "original" or gt_setting == "z-lock"):
        aruco_dir = f'{video_dir}/aruco_original.json'
    elif(gt_setting == "rot" or gt_setting == "both"):
        aruco_dir = f'{video_dir}/aruco_locked.json'
    else:
        print("ERROR!")

    with open(aruco_dir) as file:
        aruco_file = json.load(file)

    # Iterate over each item
    aruco_results = []
    finger_aruco = []
    for frame,poselist in aruco_file.items():
        for pose in poselist:
            if (pose['aruco_id'] == aruco_id):
                aruco_results.append(pose['tmat'])
            if (pose['aruco_id'] == aruco_id_finger):
                finger_aruco.append(pose['tmat'])

    return aruco_results, finger_aruco

def print_metrics(add_list, add_mod_list, model_pts):
    #################################
    #Define thresholds for evaluation (from 0 to 10% of object diameter)
    thresholds_full = np.linspace(0, 0.1, 100) # This should be 0 - 0.1 Meters (10 cm)

    # Calculate accuracy curve and AUC
    accuracies = compute_accuracy_curve_percent(add_list, thresholds_full)
    auc_value = compute_auc_percent(accuracies, thresholds_full)

    mod_accuracies = compute_accuracy_curve_percent(add_mod_list, thresholds_full)
    mod_auc_value = compute_auc_percent(mod_accuracies, thresholds_full)

    # Calculate accuracy at specific thresholds
    diameter = np.max(np.linalg.norm(model_pts[:, None] - model_pts[None, :], axis=-1))
    threshold_5_percent = 0.05 * diameter
    threshold_10_percent = 0.1 * diameter
    accuracy_5_percent = np.mean(np.array(add_list) <= threshold_5_percent) * 100.0  # Convert to percentage
    accuracy_10_percent = np.mean(np.array(add_list) <= threshold_10_percent) * 100.0  # Convert to percentage
    
    mod_accuracy_5_percent = np.mean(np.array(add_mod_list) <= threshold_5_percent) * 100.0  # Convert to percentage
    mod_accuracy_10_percent = np.mean(np.array(add_mod_list) <= threshold_10_percent) * 100.0  # Convert to percentage

    # Print evaluation results
    print(f"\nEvaluation Results:")
    print(f"Number of frames: {len(add_list)}")
    print(f"Mean ADD error: {np.mean(add_list):.6f}")
    print(f"Median ADD error: {np.median(add_list):.6f}")
    print(f"Min ADD error: {np.min(add_list):.6f}")
    print(f"Max ADD error: {np.max(add_list):.6f}")
    print("FoundationPose Results:")
    print(f"AUC value: {auc_value:.2f}% (area under recall-threshold curve)")
    print(f"10% diameter threshold ({threshold_10_percent:.6f}): {accuracy_10_percent:.2f}% accuracy")
    print(f"5% diameter threshold ({threshold_5_percent:.6f}): {accuracy_5_percent:.2f}% accuracy")
    print("Modified Results:")
    print(f"mod AUC value: {mod_auc_value:.2f}% (area under recall-threshold curve)")
    print(f"mod 10% diameter threshold {mod_accuracy_10_percent:.2f}% accuracy")
    print(f"mod 5% diameter threshold {mod_accuracy_5_percent:.2f}% accuracy")

def main(video_dir, shapename, gt_setting, mod_setting, plot_run, visualize, track_collision, channel):

    # Load object file
    if(shapename == "smallhex"):
        mesh_path = ("/media/bella/bellssd2/FoundationPose/org_tests/mesh/smallhex_centered_detailed.obj")
    if(shapename == "bighex"):
        mesh_path = ("/media/bella/bellssd2/FoundationPose/org_tests/mesh/bighex_centered_detailed.obj")

    cube_mesh = trimesh.load(mesh_path, force="mesh")
    model_pts = cube_mesh.vertices
    to_origin, extents = trimesh.bounds.oriented_bounds(cube_mesh)

    finger_mesh = trimesh.load("/media/bella/bellssd2/FoundationPose/org_tests/mesh/bell_v2_centered.obj", force='mesh')

    object_poses, hand_poses = read_fp_results(video_dir)
    aruco_results, finger_aruco = read_aruco_results(shapename, gt_setting, video_dir)

    if(track_collision):
        collision_manager = CollisionManager(channel)

    if(visualize):
        scene_viewer = CollisionViewer(channel)

    #**************************

    def make_tracked_object(mesh, object_id, channel, name):
        obj = TrackedObject(mesh, object_id, channel, name)
        if(track_collision):
            obj.slice_self(4, 4, 4)
            collision_manager.register_object(obj)
        return obj

    tracked_cube_fp = make_tracked_object(cube_mesh, 1, "FP", 'cube_fp')
    tracked_cube_mod = make_tracked_object(cube_mesh, 1, "Mod", 'cube_mod')
    tracked_cube_aruco = make_tracked_object(cube_mesh, 1, "GT", 'cube_aruco')
    tracked_finger_fp = make_tracked_object(finger_mesh, 2, "FP", 'finger_fp')
    tracked_finger_aruco = make_tracked_object(finger_mesh, 2, "Mod", 'finger_aruco')
    tracked_finger_mod = make_tracked_object(finger_mesh, 2, "GT", "finger_mod")


    add_diffs = []
    add_mod_list = []
    runs = 0
    sucesses = 0
    better_ct = 0
    gt_zpos = 0
    add_list = []
    dist_fp_fp = []
    dist_fp_gt = []
    dist_mod_mod = []
    dist_mod_gt = []
    euler_angles = []
    dist_gt_gt = []

    #**************************
    i = 0
    for obj_pose, hand_pose, aruco_result, finger_aruco_res in zip(object_poses, hand_poses, aruco_results, finger_aruco):

        # Get all object poses
        tracked_cube_fp.matrix = np.asarray(obj_pose)
        tracked_cube_mod.matrix  = np.asarray(obj_pose)
        tracked_cube_aruco.matrix  = np.asarray(aruco_result)
        tracked_finger_fp.matrix  = np.asarray(hand_pose)
        tracked_finger_mod.matrix  = np.asarray(hand_pose)
        tracked_finger_aruco.matrix  = np.asarray(finger_aruco_res)

        # Use first frame pose to correct aruco marker offset
        if(i==0): 
            offset_mat = get_correction_matrix(tracked_cube_aruco.matrix, tracked_cube_fp.matrix)
            z_lock = (tracked_cube_aruco.matrix @ offset_mat)[2, 3]
            offset_mat_finger = get_correction_matrix(tracked_finger_aruco.matrix, tracked_finger_fp.matrix)
            z_lock_finger = (tracked_finger_aruco.matrix @ offset_mat_finger)[2, 3]
        
        tracked_cube_aruco.matrix = tracked_cube_aruco.matrix @ offset_mat
        tracked_finger_aruco.matrix = tracked_finger_aruco.matrix  @ offset_mat_finger
        if(gt_setting == "z-lock" or gt_setting == "both"):
            tracked_cube_aruco.matrix[2,3] = z_lock
            tracked_finger_aruco.matrix[2,3] = z_lock_finger

        # Change z to maintain same value as first prediction
        if(i==0):
            (_, _, gt_zpos, gt_rx, gt_ry, _), _ = get_transform_params(tracked_cube_fp)
            gt_zpos_finger = tracked_finger_fp.matrix[2, 3]
            R_fixed = tracked_cube_mod.matrix[:3, :3]
            R_fixed_finger = tracked_finger_mod.matrix[:3, :3]
        else:
            # Set z-position to match the first frame
            if(mod_setting != "none"):
                tracked_cube_mod.matrix[2, 3] = gt_zpos
                tracked_finger_mod.matrix[2, 3] = gt_zpos_finger

            # Plot original rotation
            if(mod_setting == "z-rot" or mod_setting == "algo"):

                euler_angles.append(R.from_matrix(tracked_cube_mod.matrix[:3, :3]).as_euler('xyz', degrees=True))

                # Unwrap Z angle to avoid sudden jump across 180/-180 boundary
                def unwrap_angle(new_angle, reference_angle):
                    delta = new_angle - reference_angle
                    if delta > 180:
                        new_angle -= 360
                    elif delta < -180:
                        new_angle += 360
                    return new_angle

                # Only update z - axis rotation for cube
                #Get yaw angle from R (rotation around Z-axis)
                R_current = tracked_cube_mod.matrix[:3, :3]
                # Convert both to Euler angles (XYZ order)
                euler_init = R.from_matrix(R_fixed).as_euler('xyz', degrees=False)
                euler_curr = R.from_matrix(R_current).as_euler('xyz', degrees=False)
                # Preserve X and Y from initial, replace Z (yaw) from current
                euler_init[2] = unwrap_angle(euler_curr[2], euler_init[2])
                # Convert back to rotation matrix
                R_updated = R.from_euler('xyz', euler_init).as_matrix()
                # Update the tracked object rotation matrix
                tracked_cube_mod.matrix[:3, :3] = R_updated
            
        # Update scene viewer if using
        if(i==0):
            if(visualize):
                scene_viewer.add_object_to_scene(tracked_cube_fp)
                scene_viewer.add_object_to_scene(tracked_cube_aruco)
                scene_viewer.add_object_to_scene(tracked_cube_mod)
                scene_viewer.add_object_to_scene(tracked_finger_fp)
                scene_viewer.add_object_to_scene(tracked_finger_aruco)
                scene_viewer.add_object_to_scene(tracked_finger_mod)
                scene_viewer.publish_scene(run_as_live=True)
        else:
            if(track_collision):
                collision_manager.reset_collision()
                collision_manager.update_poses()
                collision_manager.channel_collision_check()
            if(visualize):
                scene_viewer.update_objects_in_scene()
            if(track_collision):
                scene_viewer.update_all_colors()

        # Get ADD measurements for Foundation Pose
        pred_rot, pred_trans = rot_trans_from_mat(tracked_cube_fp.matrix)
        gt_rot, gt_trans = rot_trans_from_mat(tracked_cube_aruco.matrix)

        original_add = calculate_add_error(model_pts, pred_rot, pred_trans, gt_rot, gt_trans)
        print("ADD error:", original_add)
        add_list.append(original_add)

        # Get ADD measurements for Z-Locked Modification
        pred_rot_mod, pred_trans_mod = rot_trans_from_mat(tracked_cube_mod.matrix)
        mod_add = calculate_add_error(model_pts, pred_rot_mod, pred_trans_mod, gt_rot, gt_trans)
        add_mod_list.append(mod_add)

        ######################################################################################################
        
        # Distance measurements
        pairs = [
            (tracked_cube_fp, tracked_finger_fp, dist_fp_fp),
            (tracked_cube_fp, tracked_finger_aruco, dist_fp_gt),
            (tracked_cube_mod, tracked_finger_mod, dist_mod_mod),
            (tracked_cube_mod, tracked_finger_aruco, dist_mod_gt),
            (tracked_cube_aruco, tracked_finger_aruco, dist_gt_gt),
        ]

        for obj1, obj2, dist_list in pairs:
            dist = tracked_obj_dist_check(obj1, obj2)
            is_col, col_dist, _ = tracked_obj_collision_check(obj1, obj2)
            dist_list.append(dist if dist > 0 else col_dist)
                            
        ######################################################################################################

        #if this_dist <= 0 or this_dist > 0.02:
        #if this_col >= 0.047 or this_dist > 0.02:
            # 0.0035, 0.005
        dist = tracked_obj_dist_check(tracked_cube_mod, tracked_finger_mod)
        is_col, col_dist, _ = tracked_obj_collision_check(tracked_cube_mod, obj2)
        
        if(mod_setting == "algo" and (dist >= 0.0065 or col_dist > 0.006)):

            print("Algrithm triggered on frame: ", i)
            print("Distance: ", mod_mod_dist)
            print("Collision :", mod_mod_col)
            runs += 1

            cube_params, cube_mat = get_transform_params(tracked_cube_mod)
            finger_params, finger_mat = get_transform_params(tracked_finger_mod)

            # Optimize the transformation to minimize the loss function -- try method='Nelder-Mead'
            arguements = (tracked_cube_mod, tracked_finger_mod)

            constraints = {'type': 'eq', 'fun': equality_constraint, 'args': arguements}
            trans_max = extents[0]/10 #8 #10 # 20
            tz_max = extents[0]/15 #20
            rot_max = np.pi/20 # 10 # 15
            tx, ty, tz, rx, ry, rz = cube_params[:6]

            bounds = ((tx-trans_max, tx+trans_max),
                    (ty-trans_max, ty+trans_max),
                    (tz-tz_max, tz+tz_max),
                    (rx-rot_max, rx+rot_max),
                    (ry-rot_max, ry+rot_max),
                    (rz-rot_max, rz+rot_max))
            
            # Set Tz to the inital z position altogether
            def CB(x):
                rotvec = np.array([rx, ry, rz])
                rotation = R.from_rotvec(rotvec).as_matrix()
                adj_pose = np.eye(4)
                adj_pose[:3, :3] = rotation
                adj_pose[:3, 3] = [tx, ty, tz]
                tracked_cube_mod.matrix = adj_pose
                if visualize:
                    scene_viewer.update_object_pose(tracked_cube_mod)

                #print("In callback")
                #tx, ty, tz, rx, ry, rz = x
                #translation = trimesh.transformations.translation_matrix([tx, ty, tz])
                #rotation = trimesh.transformations.euler_matrix(rx, ry, rz, 'sxyz')
                #adj_pose = np.dot(translation, rotation)
                #tracked_cube_mod.matrix = adj_pose
                #scene_viewer.update_object_pose(tracked_cube_mod)

            #SLSQP
            result = minimize(loss_from_init, cube_params, method='SLSQP', args=arguements, callback=CB,
                              bounds=bounds, constraints=constraints, options={'disp': True, 'maxiter': 50}) #200

            if(result.success):
                sucesses += 1

                # Extract optimized transformation parameters
                optimized_params = result.x
                tx, ty, tz, rx, ry, rz = optimized_params
                rotvec = np.array([rx, ry, rz])
                if np.linalg.norm(rotvec) > np.pi:
                    rotvec = -rotvec

                # Create the final transformation matrix
                translation = trimesh.transformations.translation_matrix([tx, ty, tz])
                #rotation = trimesh.transformations.euler_matrix(rx, ry, rz, 'sxyz')
                rotation = R.from_rotvec(rotvec).as_matrix()
                #adj_pose = np.dot(translation, rotation)
                adj_pose = np.eye(4)
                adj_pose[:3, :3] = rotation
                adj_pose[:3, 3] = [tx, ty, tz]

                # Apply the optimized transformation to the mesh
                tracked_cube_mod.matrix = adj_pose
                scene_viewer.update_object_pose(tracked_cube_mod)

                # Get new collision
                this_dist = tracked_obj_dist_check(tracked_cube_mod, tracked_finger_mod)
                is_col, this_col, this_norm = tracked_obj_collision_check(tracked_cube_mod, tracked_finger_mod)
                print("Algrithm ran sucessfully on frame: ", i)
                print("Distance: ", this_dist)
                print("Collision :", this_col)
                
                # Get new distance
                if(this_dist > 0):
                    dist_mod_mod[i] = (this_dist)
                else:
                    dist_mod_mod[i] = (this_col)

                # Get new ADD
                pred_rot, pred_trans = rot_trans_from_mat(tracked_cube_mod.matrix)
                mod_add = calculate_add_error(model_pts, pred_rot, pred_trans, gt_rot, gt_trans)
                add_mod_list[i] = mod_add
                add_diff = original_add - mod_add
                print("Change ADD:", original_add - mod_add)
                add_diffs.append(add_diff)
                if(add_diff > 0):
                    better_ct += 1

        i += 1
        #time.sleep(0.1)

    if(mod_setting == "algo"):
        print("Total frames: ", i)
        print("Total runs: ", runs)
        print("Total sucesses: ", sucesses)
        print("Better ct:", better_ct)
        print("Avg diff:", np.mean(add_diffs))
        print(add_diffs)

    ######################################################################################################

    output_dir = f"{video_dir}/offline"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"FP-FP Average Dist: {np.mean(dist_fp_fp):.6f}")
    print(f"FP-GT Average Dist: {np.mean(dist_fp_gt):.6f}")
    print(f"Mod-Mod Average Dist: {np.mean(dist_mod_mod):.6f}")
    print(f"Mod-GT Average Dist: {np.mean(dist_mod_gt):.6f}")
    print(f"GT-GT Average Dist: {np.mean(dist_gt_gt):.6f}")

    print_metrics(add_list, add_mod_list, model_pts)

    # Visualize results
    if(plot_run):
        # 0. Plot Rotation using euler angles
        plt.figure(figsize=(10, 5))
        euler_angles = np.array(euler_angles)  # Shape: (num_frames, 3)
        timescale = np.arange(len(euler_angles))
        plt.plot(timescale, euler_angles[:, 0], label='X (Roll)', color='r')
        plt.plot(timescale, euler_angles[:, 1], label='Y (Pitch)', color='g')
        plt.plot(timescale, euler_angles[:, 2], label='Z (Yaw)', color='b')
        plt.xlabel('Time')
        plt.ylabel('Rotation (degrees)')
        plt.title('Rotation over Time (Euler Angles)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 1. Plot accuracy curve
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds_full, mod_accuracies)
        plt.xlabel('ADD Threshold (M)')
        plt.ylabel('Accuracy')
        plt.title(f'ADD Accuracy Curve (AUC: {mod_auc_value:.2f}%, 10% diameter accuracy: {mod_accuracy_10_percent:.2f}%)')
        plt.axvline(x=threshold_10_percent, color='r', linestyle='--', label='10% Diameter')
        plt.grid(True)
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'add_accuracy_curve.png'))
        plt.show()

        # 2. Plot ADD error distribution
        plt.figure(figsize=(10, 6))
        plt.hist(add_mod_list, bins=30, alpha=0.75)
        plt.axvline(x=threshold_5_percent, color='g', linestyle='--', label='5% Diameter')
        plt.axvline(x=threshold_10_percent, color='r', linestyle='--', label='10% Diameter')
        plt.xlabel('ADD Error')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Modified ADD Error Distribution')
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'mod_add_error_distribution.png'))
        plt.show()

        # 2. Plot ADD error distribution
        plt.figure(figsize=(10, 6))
        plt.hist(add_list, bins=30, alpha=0.75)
        plt.axvline(x=threshold_5_percent, color='g', linestyle='--', label='5% Diameter')
        plt.axvline(x=threshold_10_percent, color='r', linestyle='--', label='10% Diameter')
        plt.xlabel('ADD Error')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('FP ADD Error Distribution')
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'fp_add_error_distribution.png'))
        plt.show()

        # 3. Plot ADD error over time
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(add_list)), add_list, label="FP ADD", color='blue')
        plt.plot(range(len(add_mod_list)), add_mod_list, label='Mod ADD', color='orange')
        plt.axhline(y=threshold_5_percent, color='g', linestyle='--', label='5% Diameter')
        plt.axhline(y=threshold_10_percent, color='r', linestyle='--', label='10% Diameter')
        plt.xlabel('Frame')
        plt.ylabel('ADD Error')
        plt.legend()
        plt.title('ADD Error Over Time')
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'add_error_over_time.png'))
        plt.show()

        ##########################################################################
        # Save results to JSON files
        # output_files = {
        #     "fp_results_finger.json": finger_results,
        #     "fp_results_cube_o.json": cube_results_o,
        #     "fp_results_cube_m.json": cube_results_m,
        #     "metrics.json": frames_algo_runs
        # }

        # for filename, data in output_files.items():
        #     output_path = os.path.join(output_dir, filename)
        #     with open(output_path, "w") as f:
        #         json.dump(data, f, indent=4)

#####################################################################################################################

# SET PARAMETERS

folder = "/media/bella/bellssd2/FoundationPose/org_tests/Aruco/BigHexagon_Rotate_Normal_Aruco"
shape = "bighex" # can be "smallhex" or "bighex"
gt_setting = "z-lock" # Can be "original", "z-lock", "rot", or "both"
mod_setting = "z-rot" # Can be "none", "z", "z-rot", "algo"
plot_run = False
visualize = True
track_collison = True
channel = "Algo" # Used for viz and collision, can be "FP", "Mod", "GT", otherwise defaults to all

main(folder, shape, gt_setting, mod_setting, plot_run, visualize, track_collison, channel)




