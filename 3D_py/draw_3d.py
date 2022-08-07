import numpy as np
import open3d as o3d
from kitti_Dataset import *
#参考网址:https://blog.csdn.net/weixin_44491667/article/details/120960701

# 根据偏航角计算旋转矩阵（逆时针旋转）
def rot_y(rotation_y):
    cos = np.cos(rotation_y)
    sin = np.sin(rotation_y)
    R = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
    return R

#绘制线框
def draw_3dframeworks(vis,points):
    position = points
    points_box = np.transpose(position)
    lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                          [0, 4], [1, 5], [2, 6], [3, 7], [0, 5], [1, 4]])
    #线框的颜色,(1,0,0)红色,暂定颜色为 [247 / 255., 8 / 255., 255 / 255.]
    # colors = np.array([[247/255., 8/255., 255/255.] for j in range(len(lines_box))])#紫红色
    # colors = np.array([[230 / 255., 146 / 255., 56 / 255.] for j in range(len(lines_box))])#棕色
    colors = np.array([[250 / 255., 115 / 255., 12 / 255.] for j in range(len(lines_box))])#橙色


    # 定义box的连接线
    line_set = o3d.geometry.LineSet()
    # 把八个顶点的空间信息转换成o3d可以使用的数据类型
    line_set.points = o3d.utility.Vector3dVector(points_box)
    # 将八个顶点连接次序的信息转换成o3d可以使用的数据类型
    line_set.lines = o3d.utility.Vector2iVector(lines_box)
    # 设置每条线段的颜色
    line_set.colors = o3d.utility.Vector3dVector(colors)

    vis.add_geometry(line_set)#将线框加入点云数据
    # vis.update_renderer()  # 可视化器渲染新的一帧



if __name__ == "__main__":
    dir_path ="./kitti/"
    index = 10  #图片的标号
    split = "training"
    dataset = Kitti_Dataset(dir_path, split=split)
    obj = dataset.get_labels(index)#标签
    print("label",obj[0].rotation_y)
    img3_d = dataset.get_rgb(index)#图片
    calib1 = dataset.get_calib(index)#点云
    pc = dataset.get_pcs(index)
    print(img3_d.shape)

    #创建PointCloud数据
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)
    point_cloud.paint_uniform_color([0/255, 0/255, 0/255])#点云的颜色

    vis =o3d.visualization.Visualizer()
    vis.create_window(width=327 ,height=194)#创建窗口

    # 渲染配置
    render_option = vis.get_render_option()
    render_option.background_color = np.array([1, 1, 1])# 设置背景颜色
    render_option.point_size = 0.01  # 设置渲染点的大小,点云的体素大小，可以影响点云的粗细

    vis.add_geometry(point_cloud)#添加点云


    for obj_index in range(len(obj)):
        if obj[obj_index].name == "Car" or obj[obj_index].name == "Pedestrian" or obj[obj_index].name == "Cyclist":
            # 阈值设置 ioc 
            # 如果需要显示自己的trainninglabel结果，需要取消这样的注释，并取消object3d.py最后一行的注释
            #if (obj[obj_index].name == "Car" and obj[obj_index].ioc >= 0.7) or  obj[obj_index].ioc > 0.5:
            R = rot_y(obj[obj_index].rotation_y)
            h, w, l = obj[obj_index].dimensions[0], obj[obj_index].dimensions[1], obj[obj_index].dimensions[2]
            x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
            y = [0, 0, 0, 0, -h, -h, -h, -h]
            # y = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
            z = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
            # 得到目标物体经过旋转之后的实际尺寸（得到其在相机坐标系下的实际尺寸）
            corner_3d = np.vstack([x, y, z])
            corner_3d = np.dot(R, corner_3d)

            # 将该物体移动到相机坐标系下的原点处（涉及到坐标的移动，直接相加就行）
            corner_3d[0, :] += obj[obj_index].location[0]
            corner_3d[1, :] += obj[obj_index].location[1]
            corner_3d[2, :] += obj[obj_index].location[2]
            corner_3d = np.vstack((corner_3d, np.zeros((1, corner_3d.shape[-1]))))
            corner_3d[-1][-1] = 1

            inv_Tr = np.zeros_like(calib1.Tr_velo_to_cam)
            inv_Tr[0:3, 0:3] = np.transpose(calib1.Tr_velo_to_cam[0:3, 0:3])
            inv_Tr[0:3, 3] = np.dot(-np.transpose(calib1.Tr_velo_to_cam[0:3, 0:3]), calib1.Tr_velo_to_cam[0:3, 3])

            Y = np.dot(inv_Tr, corner_3d)
            draw_3dframeworks(vis, Y)

    #函数视角配置
    # view_control = vis.get_view_control()
    # #垂直指向屏幕外的向量，三维空间中有无数向量，垂直指向屏幕外的只有一个
    # view_control.set_front([0, 0, 1])
    # # 拖动模型旋转时，围绕哪个点进行旋转
    # view_control.set_lookat([0, 0, 0])
    # #是设置指向屏幕上方的向量，当设置了垂直指向屏幕外的向量后，
    # # 模型三维空间中的哪个面和屏幕平行就确定了（垂直屏幕的向量相当于法向量）
    # view_control.set_up([1, 0, 0])
    # view_control.translate(0, 0)#平移
    # view_control.set_zoom(0.2)#小数放大，整数缩小

    #json视角配置，通过鼠标调整视角以后，按下 p 就会截屏且保存一个.json文件，这个json文件里保存着视角配置.
    view_control = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters('view_small.json')
    view_control.convert_from_pinhole_camera_parameters(param)

    vis.run()