# run in conda env pcdet
import os
import argparse
import numpy as np

# import open3d
import mayavi.mlab as mlab
from pcdet.utils import calibration_kitti, box_utils
from visual_utils import visualize_utils as V

DATA_PATH = '/home/roy-lnx/data/KITTI/object/testing'
RESULT_PATH = '/home/roy-lnx/Desktop/test_results/1_v1_flip_posft_02width_3R_lv23_lr007_82db_mod_run2'

BOX_COLORMAP = {
    'Car': (1.0, 0, 0),
    'Pedestrian': (0, 1.0, 0),
    'Cyclist': (0, 0, 1.0),
    'Sample_A': (1.0, 1.0, 0),
    'Sample_B': (1.0, 0, 1.0),
    'Sample_C': (0, 1.0, 1.0)}

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--frame_id', type=int, default='0',
                        help='specify the frame for demo')
    parser.add_argument('--result_path', type=str, default='/home/roy-lnx/Desktop/test_results/1_v1_flip_posft_02width_3R_lv23_lr007_82db_mod_run2',
                        help='specify the result file or directory')

    args = parser.parse_args()

    return args

def draw_points(fig, pts, show_intensity=False, draw_origin=True):

    if show_intensity:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    else:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    if draw_origin:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
        mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), tube_radius=0.1)

    return fig

def draw_grid(x1, y1, x2, y2, fig, tube_radius=None, color=(0.5, 0.5, 0.5)):
    mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    return fig


def draw_multi_grid_range(fig, grid_size=5, bv_range=(0, -40, 80, 40)):
    for x in range(bv_range[0], bv_range[2], grid_size):
        for y in range(bv_range[1], bv_range[3], grid_size):
            fig = draw_grid(x, y, x + grid_size, y + grid_size, fig)

    return fig

def draw_boxes(fig, corners3d, pred_classes=None, scores=None, line_width=2, tube_radius=None, use_color=True):

    for n in range(corners3d.shape[0]):

        b = corners3d[n]  # (7, 3)

        if use_color:
            if pred_classes is not None:
                if len(pred_classes) == 1:
                    color = BOX_COLORMAP[pred_classes[0]]
                else:
                    color = BOX_COLORMAP[pred_classes[n]]
            else:
                color = (1.0, 1.0, 0)

            if scores is not None:
                if isinstance(scores, np.ndarray):
                    mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%.2f' % scores[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)
                else:
                    mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%.2f' % scores[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)

        else:
            color = (1.0, 0, 0)
            mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%s' % pred_classes[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)

        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k, k + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

        i, j = 0, 5
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)
        i, j = 1, 4
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)

    return fig


def get_boxes_lidar(frame_id, preds):
	dims = preds[:, 5:8]
	permute = [2, 0, 1]
	dims = dims[:, permute]

	locs = preds[:, 8:11]
	rot = preds[:, 11:12]
	scores = preds[:, -1]
	pred_boxes_camera = np.concatenate([locs, dims, rot], axis=1)


	calib_file = os.path.join(DATA_PATH, 'calib/%06d.txt' % frame_id)
	calib = calibration_kitti.Calibration(calib_file)
	pred_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(pred_boxes_camera, calib)

	return pred_boxes_lidar


def draw_2d_scene():
	pass

def draw_3d_scene(DATA_PATH, frame_id, preds, pred_classes):
    pc = np.fromfile(os.path.join(DATA_PATH, 'velodyne/%06d.bin' % frame_id), dtype=np.float32).reshape(-1, 4)
    print('FRAME ID: %06d' %  frame_id)


    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0), engine=None, size=(1000, 1200))

    fig = draw_points(fig, pc)
    fig = draw_multi_grid_range(fig, grid_size=20, bv_range=(0, -40, 80, 40))

    pred_boxes_lidar = get_boxes_lidar(frame_id, preds)
    corners3d = V.boxes_to_corners_3d(pred_boxes_lidar)

    fig = draw_boxes(fig, corners3d, pred_classes, preds[:, -1])

    mlab.view(azimuth=-179, elevation=54.0, distance=104.0, roll=90.0)
    return fig


def main():
    args = parse_config()

    # draw_2d_scene()
    preds = np.loadtxt(os.path.join(RESULT_PATH, '%06d.txt' % args.frame_id), delimiter=' ', usecols=np.arange(3, 16), dtype=np.float32)
    pred_classes = np.loadtxt(os.path.join(RESULT_PATH, '%06d.txt' % args.frame_id), delimiter=' ', usecols=np.arange(0, 1), dtype=str)
    if preds.ndim == 1:
        preds = preds[np.newaxis, :]
        pred_classes = pred_classes[np.newaxis]


    fig = draw_3d_scene(DATA_PATH, args.frame_id, preds, pred_classes)
    mlab.show(stop=True)



if __name__ == '__main__':
    main()
