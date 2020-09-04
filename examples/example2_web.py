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
    def __init__(self, filename_obj, front_filename_ref, right_filename_ref, left_filename_ref, top_filename_ref):
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
            if front_filename_ref:
                print("use front image")
                self.front_image_ref = scipy.misc.imread(front_filename_ref).astype('float32').mean(-1) / 255.
            else:
                self.front_image_ref = None
            if right_filename_ref:
                print("use right image")
                self.right_image_ref = scipy.misc.imread(right_filename_ref).astype('float32').mean(-1) / 255.
            else:
                self.right_image_ref = None
            if left_filename_ref:
                print("use left image")
                self.left_image_ref = scipy.misc.imread(left_filename_ref).astype('float32').mean(-1) / 255.
            else:
                self.left_image_ref = None
            if top_filename_ref:
                print("use top image")
                self.top_image_ref = scipy.misc.imread(top_filename_ref).astype('float32').mean(-1) / 255.
            else:
                self.top_image_ref = None

            # setup renderer
            renderer = neural_renderer.Renderer()
            self.renderer = renderer

            self.count = 0

    def to_gpu(self, device=None):
        super(Model, self).to_gpu(device)
        self.faces = chainer.cuda.to_gpu(self.faces, device)
        self.textures = chainer.cuda.to_gpu(self.textures, device)
        if self.front_image_ref is not None:
            self.front_image_ref = chainer.cuda.to_gpu(self.front_image_ref, device)
        if self.right_image_ref is not None:
            self.right_image_ref = chainer.cuda.to_gpu(self.right_image_ref, device)
        if self.left_image_ref is not None:
            self.left_image_ref = chainer.cuda.to_gpu(self.left_image_ref, device)
        if self.top_image_ref is not None:
            self.top_image_ref = chainer.cuda.to_gpu(self.top_image_ref, device)

    def __call__(self):
        self.renderer.viewing_angle = 30
        self.renderer.eye = neural_renderer.get_points_from_angles(2.732, 0, 0)
        front_image = self.renderer.render_silhouettes(self.vertices, self.faces)
        loss = cf.sum(cf.square(front_image - self.front_image_ref[None, :, :]))

        if self.right_image_ref is not None:
            self.renderer.eye = neural_renderer.get_points_from_angles(2.732, 0, 90)
            right_image = self.renderer.render_silhouettes(self.vertices, self.faces)
            loss = loss + cf.sum(cf.square(right_image - self.right_image_ref[None, :, :]))

        if self.left_image_ref is not None:
            self.renderer.eye = neural_renderer.get_points_from_angles(2.732, 0, 270)
            left_image = self.renderer.render_silhouettes(self.vertices, self.faces)
            loss = loss + cf.sum(cf.square(left_image - self.left_image_ref[None, :, :]))

        if self.top_image_ref is not None:
            # FIXME: if we set elevetion to 90, it renders nothing...
            self.renderer.eye = neural_renderer.get_points_from_angles(2.732, 89.9, 0)
            top_image = self.renderer.render_silhouettes(self.vertices, self.faces)
            loss = loss + cf.sum(cf.square(top_image - self.top_image_ref[None, :, :]))
        self.count = self.count + 1
        return loss

@get('/upload')
def upload():
    return '''
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="submit" value="Upload"></br>
            Front:<input type="file" name="upload-front"></br>
            Right:<input type="file" name="upload-right"></br>
            Left:<input type="file" name="upload-left"></br>
            Top:<input type="file" name="upload-top"></br>
        </form>
    '''

@route('/upload', method='POST')
def do_upload():
    global filename_obj

    upload_front = request.files.get('upload-front')
    upload_right = request.files.get('upload-right')
    upload_left = request.files.get('upload-left')
    upload_top = request.files.get('upload-top')

    #if not upload.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
    #    return 'File extension not allowed!'

    front_image_path = None
    right_image_path = None
    left_image_path = None
    top_image_path = None

    if upload_front:
        filename = upload_front.filename.lower()
        front_image_path = os.path.join('/home/poly/Downloads', filename)
        upload_front.save(front_image_path, overwrite=True)

    if upload_right:
        filename = upload_right.filename.lower()
        right_image_path = os.path.join('/home/poly/Downloads', filename)
        upload_right.save(right_image_path, overwrite=True)

    if upload_left:
        filename = upload_left.filename.lower()
        left_image_path = os.path.join('/home/poly/Downloads', filename)
        upload_left.save(left_image_path, overwrite=True)

    if upload_top:
        filename = upload_top.filename.lower()
        top_image_path = os.path.join('/home/poly/Downloads', filename)
        upload_top.save(top_image_path, overwrite=True)

    model = Model(filename_obj, front_image_path, right_image_path, left_image_path, top_image_path)
    model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    def worker():
        loop = tqdm.tqdm(range(1000))
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
                str_list.append("{{\"x\":{:.6f},\"y\":{:.6f},\"z\":{:.6f}}}".format(v[0], v[1], v[2]))
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
