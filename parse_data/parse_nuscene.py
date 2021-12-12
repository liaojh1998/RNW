from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from numba import njit
import numpy as np
import re
import os
from matplotlib import pyplot as plt
import pickle

from tqdm import tqdm


_FULL_SIZE = (384, 192)


def get_samples_list(nusc, first_sample_token, last_sample_token, nbr_samples):
    """
    given data directory and video clip number,
    """
    sample = nusc.get('sample', first_sample_token)
    assert sample["prev"] == "", "first sample should be the first frame of the video clip"
    samples_list = [sample]
    for _ in range(1, nbr_samples):
        sample = nusc.get('sample', sample['next'])
        samples_list.append(sample)
    assert sample["next"] == "" and sample["token"] == last_sample_token, "did not reach last sample according to scene info"
    return samples_list

def get_files_of_samples_list(nusc, samples_list, sensor):
    files = []
    for sample in samples_list:
        sample_data = nusc.get('sample_data', sample['data'][sensor])
        files.append(sample_data['filename'])
    return files

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

def parse_superpoint(out_img_data_dir, out_superpoint_dir):
    cwd = os.getcwd()
    out_img_data_dir = os.path.join(cwd, out_img_data_dir)
    out_superpoint_dir = os.path.join(cwd, out_superpoint_dir)
    os.system(
        """
        python3 SuperPointPretrainedNetwork/demo_superpoint.py\
            {} \
            --weights_path SuperPointPretrainedNetwork/superpoint_v1.pth \
            --cuda \
            --save_matches \
            --save_match_dir={} \
            --no_display
        """.format(out_img_data_dir, out_superpoint_dir)
        )

def parse_scene(nusc, scene, in_root_data_dir, out_root_data_dir, out_superpoint_dir, sensor="CAM_FRONT"):
    """
    given a scene, produce the following output to the specified output directory:
    1. create a folder under out_root_data_dir/sequence/ with name of the scene token
    2. get intrinsics and store under root/sequence/videoclipnumber/intrinsic.npy
    3. get the list of image files and store under root/sequence/videoclipnumber/image_list.txt
    4. store the name of image files under root/sequence/videoclipnumber/file_list.txt
    5. get the 2D point correspondence under superpoint_dir/videoclipnumber/img_name.p
    """
    first_sample_token = scene["first_sample_token"]
    first_sample = nusc.get('sample', first_sample_token)
    cam_front_sample_data = nusc.get('sample_data', first_sample['data'][sensor])
    # get the intrinsics
    intrinsics = get_intrinsics(nusc, cam_front_sample_data)

    # get the list of image files
    samples_list = get_samples_list(nusc, scene['first_sample_token'],
                                    scene['last_sample_token'], scene['nbr_samples'])
    image_list = get_files_of_samples_list(nusc, samples_list, cam_sensor)

    in_img_data_dir = in_root_data_dir
    out_img_data_dir = os.path.join(out_root_data_dir, "sequences", scene["token"])
    is_exist = os.path.exists(out_img_data_dir)
    if not is_exist:
        # Create a new directory because it does not exist
        os.makedirs(out_img_data_dir)
        print("Creating {}".format(out_img_data_dir))

    # stores the intrinsics
    np.save(os.path.join(out_img_data_dir, "intrinsic.npy"), intrinsics)
    # parse the image list to file
    parse_file_list(image_list, in_img_data_dir, out_img_data_dir)
    # parse superpoint
    parse_superpoint(out_img_data_dir, out_superpoint_dir)

def parse_test_samples_depth(nusc_explorer, image_list, samples_list, cam_sensor, lidar_sensor, out_gt_data_dir):
    for fn, sample in zip(image_list, samples_list):
        pt_cloud, depth = get_depth(nusc_explorer, sample, cam_sensor, lidar_sensor)
        pt_cloud = pt_cloud.T
        depth_map = np.zeros(_FULL_SIZE)
        counts = np.ones(_FULL_SIZE)
        track = set()
        for (x, y), d in zip(pt_cloud, depth):
            x = np.round(x / 1600 * _FULL_SIZE[0]).astype(np.int32)
            y = np.round(y / 1600 * _FULL_SIZE[1]).astype(np.int32)
            if x == _FULL_SIZE[0]:
                x -= 1
            if y == _FULL_SIZE[1]:
                y -= 1
            depth_map[x][y] += d
            if (x, y) in track:
                counts[x][y] += 1
            else:
                track.add((x, y))
        depth_map /= counts
        fn = fn[:-4] + ".npy"
        fn = os.path.join(out_gt_data_dir, fn)
        dn = os.path.dirname(fn)
        os.makedirs(dn, exist_ok=True)
        np.save(fn, depth_map)

def parse_test_scene(nusc_explorer, scene, cam_sensor, lidar_sensor, in_root_data_dir, out_root_data_dir, weather):
    nusc = nusc_explorer.nusc
    # get list of camera image corresponding to the scene
    samples_list = get_samples_list(nusc, scene['first_sample_token'],
                                    scene['last_sample_token'], scene['nbr_samples'])
    image_list = get_files_of_samples_list(nusc, samples_list, cam_sensor)
    # save files split
    out_img_data_dir = os.path.join(out_root_data_dir, "test", "color")
    out_gt_data_dir = os.path.join(out_root_data_dir, "test", "gt")
    print("Creating {}".format(out_img_data_dir))
    os.makedirs(out_img_data_dir, exist_ok=True)
    print("Creating {}".format(out_gt_data_dir))
    os.makedirs(out_gt_data_dir, exist_ok=True)
    # get 2d pixel level depth
    parse_test_samples_depth(nusc_explorer, image_list, samples_list, cam_sensor, lidar_sensor, out_gt_data_dir)
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
    get the depth map from the samples data and lidar
    """
    # pt cloud represents the 2d coordinate
    # depth represents the depth of the point
    # img is the camera image
    lidar_sample = nusc_explorer.nusc.get('sample_data', sample['data'][lidar_sensor])
    cam_sample = nusc_explorer.nusc.get('sample_data', sample['data'][cam_sensor])
    pt_cloud, depth, img = nusc_explorer.map_pointcloud_to_image(lidar_sample["token"], cam_sample["token"])
    #img.save("test.png")
    #pt_cloud = np.round(pt_cloud)
    pt_cloud = pt_cloud[:2, :]
    #pt_cloud[2, :] = depth
    #print(pt_cloud[:, 0])
    #print(depth.shape)
    #visualize_ptcloud(pt_cloud)
    return pt_cloud, depth

def format_superpoint_files(superpoint_dir):
    pts = []
    for file in os.listdir(os.path.join(os.getcwd(), superpoint_dir)):
        fn = os.fsdecode(file)
        fn = os.path.join(os.getcwd(), superpoint_dir, fn)
        print(fn)
        if fn.endswith(".p"):
            # print(os.path.join(directory, filename))
            with open(fn, "rb") as f:
                pt = pickle.load(f)
                pts.append((fn, pt))
    # get the max size of pts
    max_len = max([len(pt) for fn, pt in pts])
    print(max_len)
    new_pts = np.zeros((len(pts), max_len, 5)).astype(np.int32)
    for i, (fn, pt) in enumerate(pts):
        new_pts[i, :len(pt), :] = pt
    # save converted array to file
    for (fn, _), pt in zip(pts, new_pts):
        with open(fn, "wb") as f:
            pickle.dump(pt, f)
    return

    # reschedule
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
    for scene_token in tqdm(test_scene_ids["day_scene"]):
        weather = "day"
        scene = nusc.get('scene', scene_token)
        parse_test_scene(nusc_explorer, scene, cam_sensor, lidar_sensor, DATA_DIR, OUT_DATA_DIR, weather)
    for scene_token in tqdm(test_scene_ids["night_scene"]):
        weather = "night"
        scene = nusc.get('scene', scene_token)
        parse_test_scene(nusc_explorer, scene, cam_sensor, lidar_sensor, DATA_DIR, OUT_DATA_DIR, weather)

def get_train_data():
    # directory where to read the raw data from nuscene
    DATA_DIR = "data/nuscene/mini"
    # directory where to save the parsed data from nuscene
    OUT_DATA_DIR = "data/parsed_nuscene/mini"
    # directory where to save superpoint point correspondence file
    SUPERPOINT_DIR = "data/parsed_nuscene/mini_correspondence"
    sensor = "CAM_FRONT"
    nusc = NuScenes(version='v1.0-mini', dataroot=DATA_DIR, verbose=True)

    for scene in nusc.scene:
        parse_scene(nusc, scene, DATA_DIR, OUT_DATA_DIR, SUPERPOINT_DIR, sensor)

    # make sure all superpoint is parsed:
    format_superpoint_files(SUPERPOINT_DIR)

def main():
    #get_test_data()
    get_train_data()


if __name__ == "__main__":
    #test()
    main()
