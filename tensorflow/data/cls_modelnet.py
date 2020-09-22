import os
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, required=True,
                    help='The command to run.')
parser.add_argument('--converter', type=str, required=False,
                    default='util/convert_tfrecords.py',
                    help='The path of the convert_tfrecords')
parser.add_argument('--scanner', type=str,  required=False,
                    help='The path of the virtual_scanner')
parser.add_argument('--octree', type=str, required=False, default='octree',
                    help='The path of the octree')
parser.add_argument('--simplify_points', type=str, required=False,
                    default='simplify_points',
                    help='The path of the simplify_points')


abs_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
root_folder = os.path.join(abs_path, 'script/dataset/ModelNet40')


args = parser.parse_args()
octree = args.octree
converter = os.path.join(abs_path, args.converter)
virtual_scanner = args.scanner
simplify = args.simplify_points


def download_m40():
  # download via wget
  if not os.path.exists(root_folder):
    os.makedirs(root_folder)
  url = 'http://modelnet.cs.princeton.edu/ModelNet40.zip'
  cmd = 'wget %s -O %s/ModelNet40.zip' % (url, root_folder)
  print(cmd)
  os.system(cmd)

  # unzip
  cmd = 'unzip %s/ModelNet40.zip -d %s' % (root_folder, root_folder)
  print(cmd)
  os.system(cmd)


def download_m40_points():
    # download via wget
  if not os.path.exists(root_folder):
    os.makedirs(root_folder)
  url = 'https://www.dropbox.com/s/m233s9eza3acj2a/ModelNet40.points.zip?dl=0'
  cmd = 'wget %s -O %s/ModelNet40.points.zip' % (url, root_folder)
  print(cmd)
  os.system(cmd)

  # unzip
  cmd = 'unzip %s/ModelNet40.points.zip -d %s/ModelNet40.points' % (
      root_folder, root_folder)
  print(cmd)
  os.system(cmd)


def clean_off_file(filename):
  # read the contents of the file
  with open(filename) as fid:
    file_str = fid.read()
  # fix the file
  if file_str[0:3] != 'OFF':
    print('Error: not an OFF file: ' + filename)
  elif file_str[0:4] != 'OFF\n':
    print('Info: fix an OFF file: ' + filename)
    new_str = file_str[0:3] + '\n' + file_str[3:]
    with open(filename, 'w') as f_rewrite:
      f_rewrite.write(new_str)


def m40_get_filelist(root_folder, train=True, suffix='off'):
  filelist, category = [], []
  folders = sorted(os.listdir(root_folder))
  assert(len(folders) == 40)
  for idx, folder in enumerate(folders):
    subfolder = 'train' if train else 'test'
    current_folder = os.path.join(root_folder, folder, subfolder)
    filenames = sorted(os.listdir(current_folder))
    for filename in filenames:
      if filename.endswith(suffix):
        filelist.append(os.path.join(folder, subfolder, filename))
        category.append(idx)
  return filelist, category


def m40_move_files(src_folder, des_folder, suffix):
  folders = os.listdir(src_folder)
  for folder in folders:
    for subfolder in ['train', 'test']:
      curr_src_folder = os.path.join(src_folder, folder, subfolder)
      curr_des_folder = os.path.join(des_folder, folder, subfolder)
      if not os.path.exists(curr_des_folder):
        os.makedirs(curr_des_folder)
      filenames = os.listdir(curr_src_folder)
      for filename in filenames:
        if filename.endswith('.points'):
          os.rename(os.path.join(curr_src_folder, filename),
                    os.path.join(curr_des_folder, filename))


def m40_convert_mesh_to_points():
  mesh_folder = os.path.join(root_folder, 'ModelNet40')
  # Delete 3 files since the virtualscanner can not well deal with them
  filelist = ['cone/train/cone_0117.off',
              'curtain/train/curtain_0066.off',
              'car/train/car_0021.off.off']
  for filename in filelist:
    filename = os.path.join(mesh_folder, filename)
    if os.path.exists(filename):
      os.remove(filename)

  # clean the off files
  train_list, _ = m40_get_filelist(mesh_folder, train=True,  suffix='off')
  test_list, _ = m40_get_filelist(mesh_folder, train=False, suffix='off')
  filelist = train_list + test_list
  for filename in filelist:
    clean_off_file(os.path.join(mesh_folder, filename))

  # run virtualscanner
  folders = os.listdir(mesh_folder)
  for folder in folders:
    for subfolder in ['train', 'test']:
      curr_folder = os.path.join(mesh_folder, folder, subfolder)
      cmd = '%s %s 14' % (virtual_scanner,  curr_folder)
      print(cmd)
      os.system(cmd)

  # move points
  m40_move_files(mesh_folder, mesh_folder + '.points', 'points')


def m40_convert_points_to_octree(depth=5, adaptive=0, node_dis=0):
  points_folder = os.path.join(root_folder, 'ModelNet40.points')
  folders = os.listdir(points_folder)
  for folder in folders:
    for subfolder in ['train', 'test']:
      curr_folder = os.path.join(points_folder, folder, subfolder)
      # write filelist to disk
      filenames = os.listdir(curr_folder)
      filelist_name = os.path.join(curr_folder, 'list.txt')
      with open(filelist_name, 'w') as fid:
        for i, filename in enumerate(filenames):
          if filename.endswith('.points') and i % stride == 0:
            fid.write(os.path.join(curr_folder, filename) + '\n')
      # run octree
      octree_folder = points_folder[:-6] + 'octree.%d' % depth
      if adaptive == 1:
        octree_folder = octree_folder + '.adaptive'
      output_path = os.path.join(octree_folder, folder, subfolder)
      if not os.path.exists(output_path):
        os.makedirs(output_path)
      cmd = '%s --filenames %s --output_path %s --depth %d --adaptive %d --node_dis %d --axis z' % \
            (octree, filelist_name, output_path, depth, adaptive, node_dis)
      print(cmd)
      os.system(cmd)


def m40_simplify_points(resolution=64):
  # rename and backup the original folders
  points_folder = os.path.join(root_folder, 'ModelNet40.points')
  original_folder = points_folder + ".dense"
  if os.path.exists(points_folder):
    os.rename(points_folder, original_folder)

  folders = os.listdir(original_folder)
  for folder in folders:
    for subfolder in ['train', 'test']:
      curr_folder = os.path.join(original_folder, folder, subfolder)
      # write filelist to disk
      filenames = os.listdir(curr_folder)
      filelist_name = os.path.join(curr_folder, 'list.txt')
      with open(filelist_name, 'w') as fid:
        for filename in filenames:
          if filename.endswith('.points'):
            fid.write(os.path.join(curr_folder, filename) + '\n')
      # run simplify_points
      output_path = os.path.join(points_folder, folder, subfolder)
      if not os.path.exists(output_path):
        os.makedirs(output_path)
      cmd = '%s --filenames %s --output_path %s --dim %d' % \
            (simplify, filelist_name, output_path, resolution)
      print(cmd)
      os.system(cmd)
      os.remove(filelist_name)


def m40_generate_points_tfrecords():
  points_folder = os.path.join(root_folder, 'ModelNet40.points')
  for folder in ['train', 'test']:
    train = folder == 'train'
    shuffle = '--shuffle true' if folder == 'train' else ''
    filelist, idx = m40_get_filelist(
        points_folder, train=train, suffix='points')
    filename = os.path.join(root_folder, 'm40_%s_points_list.txt' % folder)
    with open(filename, 'w') as fid:
      for i in range(len(filelist)):
        fid.write('%s %d\n' % (filelist[i], idx[i]))
    tfrecords_name = os.path.join(
        root_folder, 'm40_%s_points.tfrecords' % folder)
    cmd = 'python %s %s --file_dir %s --list_file %s --records_name %s' % \
          (converter, shuffle, points_folder, filename, tfrecords_name)
    print(cmd)
    os.system(cmd)


def m40_generate_octree_tfrecords(depth=5):
  # generate octree
  m40_convert_points_to_octree(depth, adaptive=0, node_dis=0)

  # generate tfrecords
  octree_folder = os.path.join(root_folder, 'ModelNet40.octree.%d.adaptive' % depth)
  octree_folder = os.path.join(root_folder, 'ModelNet40_points/%s' % octree_folder_name)
  for folder in ['train', 'test']:
  #for folder in ['test']:
    train = folder == 'train'
    shuffle = '--shuffle true' if folder == 'train' else ''
    filelist, idx = m40_get_filelist(
        octree_folder, train=train, suffix='octree')
    filename = os.path.join(root_folder, 'm40_%s_octree_list.txt' % folder)
    with open(filename, 'w') as fid:
      for i in range(len(filelist)):
        fid.write('%s %d\n' % (filelist[i], idx[i]))
    tfname = os.path.join(
        root_folder, 'm40_%d_2_12_%s_octree.tfrecords' % (depth, folder))
    cmd = 'python %s %s --file_dir %s --list_file %s --records_name %s' % \
          (converter, shuffle, octree_folder, filename, tfname)
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':
  eval('%s()' % args.run)
