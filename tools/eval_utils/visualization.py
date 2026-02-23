import matplotlib.pyplot as plt  
import numpy as np  
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  
from matplotlib.animation import FuncAnimation  

def init():  
    return []  

def update(frame, faces, vertices, ax, max_frames): 
    collects = split_vfs(vertices, faces, frame, max_frames)

    mesh = Poly3DCollection(collects, alpha=0.5, edgecolor='k')  
    face_color = [0.5, 0.5, 1]  # 设置面的颜色  
    mesh.set_facecolor(face_color)  
    ax.add_collection3d(mesh)  

    azim = (frame / 5) * 20
    ax.view_init(elev=30, azim=azim)
    return []  

def create_animation(vertices, faces, max_frames=200):  
    fig = plt.figure()  
    ax = fig.add_subplot(111, projection='3d')  
    
    # 设置坐标轴范围  
    ax.set_xlim([min(vertices[:, 0]), max(vertices[:, 0])])  
    ax.set_ylim([min(vertices[:, 1]), max(vertices[:, 1])])  
    ax.set_zlim([min(vertices[:, 2]), max(vertices[:, 2])])  
    
    # 创建动画  
    anim = FuncAnimation(fig, update, frames=max_frames, init_func=init,  
                         fargs=(faces, vertices, ax, max_frames), interval=100, blit=True, repeat=False)
    return anim
    # 保存为GIF  
    # anim.save('3d_model_animation.gif', writer='pillow')  
    # plt.show()  

def split_vfs(vertices, faces, frame, max_frames):
    # 将顶点和面按max_frames来分batch
    batch = (len(faces) // max_frames) + 1
    collects = []
    # if frame < max_frames:  
    for idx in range(batch):
        face_idx = frame * batch + idx
        if face_idx < len(faces):
            face = faces[face_idx]  
            collects.append(vertices[face])
    return collects

def multi_update(frame, faces, vertices, gt_faces, gt_vertices, gt_point_cloud, ax, ax2, ax3, max_frames):  
    
    collects = split_vfs(vertices, faces, frame, max_frames)

    mesh = Poly3DCollection(collects, alpha=0.5, edgecolor='k')  
    face_color = [0.5, 0.5, 1]  # 设置面的颜色  
    mesh.set_facecolor(face_color)  
    ax.add_collection3d(mesh)  

    gt_collects = split_vfs(gt_vertices, gt_faces, frame, max_frames)
    # face = gt_faces[frame]  
    mesh = Poly3DCollection(gt_collects, alpha=0.5, edgecolor='k')  
    face_color = [1, 0.5, 0.5]  # 设置面的颜色  
    mesh.set_facecolor(face_color)  
    ax2.add_collection3d(mesh)  

    print(frame, 'fig frame')
    # 绘制点云  
    if frame < 1: 
        print(frame)

        colors = 'b'
        if gt_point_cloud.shape[1] == 6:
            normals = gt_point_cloud[:, 3:]
            normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]  
            colors = (normals + 1) / 2

        ax3.scatter(gt_point_cloud[:, 0], gt_point_cloud[:, 1], gt_point_cloud[:, 2], s=5, c=colors, marker='o')  

        # 设置坐标轴标签  
        ax3.set_xlabel('X')  
        ax3.set_ylabel('Y')  
        ax3.set_zlabel('Z')  

    azim = (frame / 5) * 20

    ax.view_init(elev=30, azim=azim)
    ax2.view_init(elev=30, azim=azim)
    ax3.view_init(elev=30, azim=azim)
    return []  



# gt_point_cloud
def create_multi_animation(vertices, faces, gt_vertices, gt_faces, gt_point_cloud=None, max_frames=200):  
    fig = plt.figure(figsize=(30, 10), dpi=50)  

    ax = fig.add_subplot(133, projection='3d')  
    # 设置坐标轴范围  
    ax.set_xlim([min(vertices[:, 0]), max(vertices[:, 0])])  
    ax.set_ylim([min(vertices[:, 1]), max(vertices[:, 1])])  
    ax.set_zlim([min(vertices[:, 2]), max(vertices[:, 2])])  


    ax2 = fig.add_subplot(132, projection='3d')  
    # 设置坐标轴范围  
    ax2.set_xlim([min(gt_vertices[:, 0]), max(gt_vertices[:, 0])])  
    ax2.set_ylim([min(gt_vertices[:, 1]), max(gt_vertices[:, 1])])  
    ax2.set_zlim([min(gt_vertices[:, 2]), max(gt_vertices[:, 2])])  


    ax3 = fig.add_subplot(131, projection='3d')  
    # 设置坐标轴范围  
    ax3.set_xlim([min(gt_point_cloud[:, 0]), max(gt_point_cloud[:, 0])])  
    ax3.set_ylim([min(gt_point_cloud[:, 1]), max(gt_point_cloud[:, 1])])  
    ax3.set_zlim([min(gt_point_cloud[:, 2]), max(gt_point_cloud[:, 2])])  

    # 创建动画  
    # frames = max(len(faces), len(gt_faces)) + 90
    frames = max_frames
    anim = FuncAnimation(fig, multi_update, frames=frames, init_func=init,  
                         fargs=(faces, vertices, gt_faces, gt_vertices, gt_point_cloud, ax, ax2, ax3, max_frames), interval=100, blit=True, repeat=False)  
    
    return anim


if __name__ == '__main__':
    import os
    import torch
    import trimesh
    from src.utils.mesh import y_up_coord_to_z_up
    
    all_generation_dir = [
        '/primexu-cfs/aigc/tmp4/meshtron/eval_output/eval_meshtron/face_1k_reso_128_model_t0.1_step17k/logs/20250305_162407/object/sample_17/meshtron_step_checkpoint-17000/seed_123',
        '/primexu-cfs/aigc/tmp4/meshtron/eval_output/eval_meshtron/face_1k_reso_128_model_t0.1_step17k/logs/20250305_162407/object/sample_18/meshtron_step_checkpoint-17000/seed_123',
        '/primexu-cfs/aigc/tmp4/meshtron/eval_output/eval_meshtron/face_1k_reso_128_model_t0.1_step17k/logs/20250305_162407/object/sample_65/meshtron_step_checkpoint-17000/seed_123',
        '/primexu-cfs/aigc/tmp4/meshtron/eval_output/eval_meshtron/face_1k_reso_128_model_t0.1_step17k/logs/20250305_162407/object/sample_83/meshtron_step_checkpoint-17000/seed_123'
    ]
    
    only_generated_res = True
    
    for idx, generation_dir in enumerate(all_generation_dir):
        gen_mesh = trimesh.load(os.path.join(generation_dir, 'generated_mesh.ply'))
        gt_mesh = trimesh.load(os.path.join(generation_dir, 'gt_mesh.ply'))
        point_cloud_shape = trimesh.load(os.path.join(generation_dir, 'point_cond.ply'))
        pc_indices = np.random.choice(len(point_cloud_shape.vertices), size=5000, replace=False)
        point_cloud = point_cloud_shape.vertices[pc_indices]
        
        if only_generated_res:
            anim = create_animation(
                vertices=y_up_coord_to_z_up(torch.tensor(gen_mesh.vertices)).numpy(), 
                faces=gen_mesh.faces,
                max_frames=200,
            )
        else:
            anim = create_animation(
                vertices=y_up_coord_to_z_up(torch.tensor(gen_mesh.vertices)).numpy(), 
                faces=gen_mesh.faces, 
                gt_vertices=y_up_coord_to_z_up(torch.tensor(gt_mesh.vertices)).numpy(), 
                gt_faces=gt_mesh.faces, 
                gt_point_cloud=y_up_coord_to_z_up(torch.tensor(point_cloud)).numpy(), 
                max_frames=200,
            )
        anim.save(f"./demo_{idx}.gif", writer='pillow')  
