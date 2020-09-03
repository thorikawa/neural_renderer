"""
Example 2. Optimizing vertices.
"""

from gevent import monkey
monkey.patch_all()
from bottle import route, run, template, request, static_file, url, get, post, response, error, abort, redirect, os

import argparse
import glob
import os
import subprocess

import chainer
import chainer.functions as cf
import numpy as np
import scipy.misc
import tqdm

import neural_renderer

filename_obj = ""

class Model(chainer.Link):
    def __init__(self, filename_obj, filename_ref):
        super(Model, self).__init__()

        with self.init_scope():
            # load .obj
            vertices, faces = neural_renderer.load_obj(filename_obj)
            self.vertices = chainer.Parameter(vertices[None, :, :])
            self.faces = faces[None, :, :]

            # create textures
            texture_size = 2
            textures = np.ones((1, self.faces.shape[1], texture_size, texture_size, texture_size, 3), 'float32')
            self.textures = textures

            # load reference image
            self.image_ref = scipy.misc.imread(filename_ref).astype('float32').mean(-1) / 255.

            # setup renderer
            renderer = neural_renderer.Renderer()
            self.renderer = renderer

    def to_gpu(self, device=None):
        super(Model, self).to_gpu(device)
        self.faces = chainer.cuda.to_gpu(self.faces, device)
        self.textures = chainer.cuda.to_gpu(self.textures, device)
        self.image_ref = chainer.cuda.to_gpu(self.image_ref, device)

    def __call__(self):
        self.renderer.eye = neural_renderer.get_points_from_angles(2.732, 0, 90)
        image = self.renderer.render_silhouettes(self.vertices, self.faces)
        loss = cf.sum(cf.square(image - self.image_ref[None, :, :]))
        return loss

@get('/upload')
def upload():
    return '''
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="submit" value="Upload"></br>
            <input type="file" name="upload"></br>
        </form>
    '''

@route('/upload', method='POST')
def do_upload():
    global filename_obj

    upload = request.files.get('upload', '')
    if not upload.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return 'File extension not allowed!'

    filename = upload.filename.lower()
    root, ext = os.path.splitext(filename)
    save_path = os.path.join('/home/poly/Downloads', filename)

    print("trying to save")
    upload.save(save_path, overwrite=True)
    print("saved input file")

    model = Model(filename_obj, save_path)
    model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    def worker():
        loop = tqdm.tqdm(range(300))
        for i in loop:
            loop.set_description('Optimizing')
            optimizer.target.cleargrads()
            loss = model()
            loss.backward()
            optimizer.update()

            str_list = []
            # to accelerate variable access, transfer the data to cpu from gpu
            model.vertices.to_cpu()
            varray = chainer.as_array(model.vertices)
            for v in varray[0]:
                str_list.append("[{0},{1},{2}]".format(v[0], v[1], v[2]))
            varrayStr = ",".join(str_list)
            model.vertices.to_gpu()
            yield "{{\"vertices\":[{0}]}}\n".format(varrayStr)
    return worker()

def main():
    global filename_obj

    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default='./examples/data/teapot.obj')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()
    filename_obj = args.filename_obj

    try:
        run(host="0.0.0.0", port=8000, server="gevent", debug=True)
    finally:
        pass

if __name__ == '__main__':
    main()
