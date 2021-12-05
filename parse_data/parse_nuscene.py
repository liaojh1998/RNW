from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from numba import njit
import numpy as np
import re
import os
from matplotlib import pyplot as plt

def get_file_list(nusc, first_cam_sample_data):
    """
    given data directory and video clip number, 
    """
    assert first_cam_sample_data["prev"] == "", "first sample data should be the first frame of the video clip"
    cam_sample_data = first_cam_sample_data
    image_name_list = []
    while True:
        # save the image file name
        #print(cam_sample_data["filename"])
        image_name_list.append(cam_sample_data["filename"])
        if cam_sample_data["next"] == "":
            break
        cam_sample_data = nusc.get('sample_data', cam_sample_data["next"])
    
    #return [cam_sample_data["filename"]]
    return image_name_list

def parse_file_list(file_list_file, in_img_data_dir, out_img_data_dir, write_txt = True):
    """
    do the following:
    1. put the list of image to outdir/file_list.txt, with name mapping
    2. link the image from the original data directory to outdir/
    """
    p = re.compile("\w+/[\w,\_]+/([\w,\_,.,\-,\+]+)")
    new_img_list_file = []
    for img_fn in file_list_file:
        new_fn = p.match(img_fn).group(1)
        new_img_list_file.append(new_fn)
    # link the image files
    for src, dst in zip(file_list_file, new_img_list_file):
        src = os.path.join(os.getcwd(), in_img_data_dir, src)
        dst = os.path.join(out_img_data_dir, dst)
        assert os.path.exists(src), "the source file does not exist"
        #print(src, dst)
        os.symlink(src, dst)
    # write the file list to the file
    if write_txt:
        dst = os.path.join(out_img_data_dir, "file_list.txt")
        with open(dst, "w") as f:
            for fn in new_img_list_file:
                f.write(fn + "\n")

def get_intrinsics(nusc, sample_data):
    """
    get the intrinsics from the specified sample_data
    """
    calibrated_sensor_token = sample_data['calibrated_sensor_token']
    cs_record = nusc.get('calibrated_sensor', calibrated_sensor_token)
    #sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    #print(cs_record)
    return np.array(cs_record["camera_intrinsic"])


def parse_scene(nusc, scene, in_root_data_dir, out_root_data_dir, sensor="CAM_FRONT"):
    """
    given a scene, produce the following output to the specified output directory:
    1. create a folder under out_root_data_dir/sequence/ with name of the scene token
    2. get intrinsics and store under root/sequence/videoclipnumber/intrinsic.npy
    3. get the list of image files and store under root/sequence/videoclipnumber/image_list.txt
    4. store the name of image files under root/sequence/videoclipnumber/file_list.txt
    """
    first_sample_token = scene["first_sample_token"]
    first_sample = nusc.get('sample', first_sample_token)
    cam_front_sample_data = nusc.get('sample_data', first_sample['data'][sensor])
    # get the intrinsics
    intrinsics = get_intrinsics(nusc, cam_front_sample_data)

    # get the list of image files
    image_list = get_file_list(nusc, cam_front_sample_data)

    in_img_data_dir = in_root_data_dir
    out_img_data_dir = os.path.join(out_root_data_dir, "sequences", scene["token"])
    is_exist = os.path.exists(out_img_data_dir)
    if not is_exist:  
        # Create a new directory because it does not exist 
        os.makedirs(out_img_data_dir)
        print("Creating {}".format(out_img_data_dir))

    # stores the intrinsics
    np.save(os.path.join(out_img_data_dir, "intrinsic.npy"), intrinsics)
    # parse the image list
    parse_file_list(image_list, in_img_data_dir, out_img_data_dir)

def parse_test_scene(nusc_explorer, scene, cam_sensor, lidar_sensor, in_root_data_dir, out_root_data_dir, weather):
    nusc = nusc_explorer.nusc
    first_sample_token = scene["first_sample_token"]
    first_sample = nusc.get('sample', first_sample_token)
    # gets 2d pixel level depth
    # TODO: loop over all the samples
    # pt_cloud, depth = get_depth(nusc_explorer, first_sample, sensor, lidar_sensor)
    # get list of camera image corresponding to the scene
    image_list = get_file_list(nusc, nusc.get('sample_data', first_sample['data'][cam_sensor]))
    out_img_data_dir = os.path.join(out_root_data_dir, "test", "color")
    is_exist = os.path.exists(out_img_data_dir)
    if not is_exist:  
        # Create a new directory because it does not exist 
        os.makedirs(out_img_data_dir)
        print("Creating {}".format(out_img_data_dir))
    #print(out_img_data_dir)
    #exit(0)
    # parse the image list
    parse_file_list(image_list, in_root_data_dir, out_img_data_dir, write_txt=False)
    # create the split file
    out_split_data_dir = os.path.join(out_root_data_dir, "splits")
    dst = os.path.join(out_split_data_dir, "{}_test_split.txt".format(weather))
    img_name_without_extension = [im[:-4] for im in image_list]
    #print(img_name_without_extension)
    with open(dst, "w") as f:
        for fn in img_name_without_extension:
            f.write(fn + "\n")

def test():
    DATA_DIR = "data/nuscene/mini"
    OUT_DATA_DIR = "data/parsed_nuscene/mini"
    sensor = "CAM_FRONT"
    nusc = NuScenes(version='v1.0-mini', dataroot=DATA_DIR, verbose=True)
    for scene in nusc.scene:
        #print(scene)
        #print(scene["token"])
        first_sample_token = scene["first_sample_token"]
        first_sample = nusc.get('sample', first_sample_token)
        print(first_sample["data"].keys())
        exit(0)
        #print(first_sample)
        cam_front_sample_data = nusc.get('sample_data', first_sample['data'][sensor])
        #print(cam_front_data)
        calibrated_sensor_token = cam_front_sample_data['calibrated_sensor_token']
        cs_record = nusc.get('calibrated_sensor', calibrated_sensor_token)
        sensor_record = nusc.get('sensor', cs_record['sensor_token'])
        print(np.array(cs_record["camera_intrinsic"]))

        print(cam_front_sample_data["filename"])
        exit(0)
    pass
    

def visualize_ptcloud(pt_cloud, h=1600, w=900):
    scale = 10
    depth_img = np.zeros((h//scale, w//scale))
    for pt in pt_cloud.T:
        x = pt[0].astype(np.int32)//scale
        y = pt[1].astype(np.int32)//scale
        depth_img[x, y] = pt[2]
        print(depth_img[x, y])
    depth_img = depth_img/np.max(depth_img)
    #print(depth_img)
    #exit(0)
    plt.imshow(depth_img.T, cmap="hot", interpolation="bilinear")
    #plt.scatter(pt_cloud[0, :], pt_cloud[1, :], c = pt_cloud[2, :], cmap='hot')
    plt.savefig("test.png")
    #plt.show()
    #print(depth_img)
    exit(0)

def get_depth(nusc_explorer, sample, cam_sensor, lidar_sensor):
    """
    get the depth map from the sample data and lidar
    """
    cam_sample_data = nusc_explorer.nusc.get('sample_data', sample['data'][cam_sensor])
    lidar_sample_data = nusc_explorer.nusc.get('sample_data', sample['data'][lidar_sensor])
    # pt cloud represents the 2d coordinate
    # depth represents the depth of the point
    # img is the camera image
    pt_cloud, depth, img =  nusc_explorer.map_pointcloud_to_image(lidar_sample_data["token"], cam_sample_data["token"])
    img.save("test.png")
    pt_cloud = np.round(pt_cloud)
    pt_cloud = pt_cloud[:2, :].astype(np.int32)
    #pt_cloud[2, :] = depth
    #print(pt_cloud[:, 0])
    #print(depth.shape)    
    #visualize_ptcloud(pt_cloud)
    return pt_cloud, depth

def get_test_scene_ids():
    day_test_scene = ["fcbccedd61424f1b85dcbf8f897f9754"]
    night_test_scene = ["e233467e827140efa4b42d2b4c435855"]
    return {"day_scene": day_test_scene, "night_scene": night_test_scene}

def get_test_data():
    DATA_DIR = "data/nuscene/mini"
    OUT_DATA_DIR = "data/parsed_nuscene/mini"
    cam_sensor = "CAM_FRONT"
    lidar_sensor = "LIDAR_TOP"
    nusc = NuScenes(version='v1.0-mini', dataroot=DATA_DIR, verbose=True)
    test_scene_ids = get_test_scene_ids()
    nusc_explorer = NuScenesExplorer(nusc)
    for scene_token in test_scene_ids["day_scene"]:  
        weather = "day"
        scene = nusc.get('scene', scene_token)
        parse_test_scene(nusc_explorer, scene, cam_sensor, lidar_sensor, DATA_DIR, OUT_DATA_DIR, weather)
    for scene_token in test_scene_ids["night_scene"]:  
        weather = "night"
        scene = nusc.get('scene', scene_token)
        parse_test_scene(nusc_explorer, scene, cam_sensor, lidar_sensor, DATA_DIR, OUT_DATA_DIR, weather)
    pass

def get_train_data():
    DATA_DIR = "data/nuscene/mini"
    OUT_DATA_DIR = "data/parsed_nuscene/mini"
    sensor = "CAM_FRONT"
    nusc = NuScenes(version='v1.0-mini', dataroot=DATA_DIR, verbose=True)
    for scene in nusc.scene:
        parse_scene(nusc, scene, DATA_DIR, OUT_DATA_DIR, sensor)

def main():
    #get_test_data()
    get_train_data()


if __name__ == "__main__":
    #test()
    main()