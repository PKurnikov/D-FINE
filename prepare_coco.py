"""Prepare MS COCO datasets"""
import os
import shutil
import argparse
import zipfile
import tqdm
import requests
import errno
import hashlib

_TARGET_DIR = os.path.expanduser('~/.encoding/data')

def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.
    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.
    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash


def mkdir(path):
    """make dir exists okay"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def download(url, path=None, overwrite=False, sha1_hash=None):
    """Download an given URL
    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split('/')[-1]
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path

    if overwrite or not os.path.exists(fname) or (sha1_hash and not check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        print('Downloading %s from %s...'%(fname, url))
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s"%url)
        total_length = r.headers.get('content-length')
        with open(fname, 'wb') as f:
            if total_length is None: # no content length header
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(r.iter_content(chunk_size=1024),
                                  total=int(total_length / 1024. + 0.5),
                                  unit='KB', unit_scale=False, dynamic_ncols=True):
                    f.write(chunk)

        if sha1_hash and not check_sha1(fname, sha1_hash):
            raise UserWarning('File {} is downloaded but the content hash does not match. ' \
                              'The repo may be outdated or download may be incomplete. ' \
                              'If the "repo_url" is overridden, consider switching to ' \
                              'the default repo.'.format(fname))

    return fname

def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize MS COCO dataset.',
        epilog='Example: python mscoco.py --download-dir ~/mscoco',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', type=str, default='/data1/Datasets/fiftyone/', help='dataset directory on disk')
    args = parser.parse_args()
    return args

def download_coco(path, overwrite=False):
    _DOWNLOAD_URLS = [
        ('http://images.cocodataset.org/zips/train2017.zip',
         '10ad623668ab00c62c096f0ed636d6aff41faca5'),
        ('http://images.cocodataset.org/zips/val2017.zip',
         '4950dc9d00dbe1c933ee0170f5797584351d2a41'),
        ('http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
         '8551ee4bb5860311e79dace7e79cb91e432e78b3'),
        #('http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip',
        # '46cdcf715b6b4f67e980b529534e79c2edffe084'),
        #('http://images.cocodataset.org/zips/test2017.zip',
        # '99813c02442f3c112d491ea6f30cecf421d0e6b3'),
        ('https://hangzh.s3.amazonaws.com/encoding/data/coco/train_ids.pth',
         '12cd266f97c8d9ea86e15a11f11bcb5faba700b6'),
        ('https://hangzh.s3.amazonaws.com/encoding/data/coco/val_ids.pth',
         '4ce037ac33cbf3712fd93280a1c5e92dae3136bb'),
    ]
    mkdir(path)
    for url, checksum in _DOWNLOAD_URLS:
        filename = download(url, path=path, overwrite=overwrite, sha1_hash=checksum)
        # extract
        if os.path.splitext(filename)[1] == '.zip':
            with zipfile.ZipFile(filename) as zf:
                zf.extractall(path=path)
        else:
            shutil.move(filename, os.path.join(path, 'annotations/'+os.path.basename(filename)))


def install_coco_api():
    repo_url = "https://github.com/cocodataset/cocoapi"
    os.system("git clone " + repo_url)
    os.system("cd cocoapi/PythonAPI/ && python setup.py install")
    shutil.rmtree('cocoapi')
    try:
        import pycocotools
    except Exception:
        print("Installing COCO API failed, please install it manually %s"%(repo_url))


if __name__ == '__main__':
    args = parse_args()
    # mkdir(os.path.expanduser('~/.encoding/data'))
    # if args.download_dir is not None:
    #     if os.path.isdir(_TARGET_DIR):
    #         os.remove(_TARGET_DIR)
    #     # make symlink
    #     os.symlink(args.download_dir, _TARGET_DIR)
    # else:
    download_coco(args.download_dir, overwrite=False)
    install_coco_api()