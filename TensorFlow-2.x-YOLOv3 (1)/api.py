import flask
from flask import  Flask, request, url_for, redirect, render_template, jsonify
import numpy as np
import cv2
import os
from detect_mnist import recognize_gj

#设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
# app.register_blueprint(gjdet)

@app.route("/")
@app.route("/index")
def index():
   return flask.render_template('index.html')

@app.route('/upload', methods=['POST', 'GET'])  # 添加路由
def upload():
    # return "ajax测试"
    if request.method == 'POST':
        f = request.files['imgFile']
 
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})
 
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
 
        upload_path = os.path.join(basepath, 'static\\uploadFiles', f.filename)  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)
 
        # 使用Opencv转换一下图片格式和名称
        img = cv2.imread(upload_path)
        cv2.imwrite(os.path.join(basepath, 'static\\uploadFiles', f.filename), img)

        # return render_template('gjdet.html',updFileName=f.filename)
 
    # return render_template('UploadFile.html')
    out_path = os.path.join(basepath, 'static\\uploadFiles\\final_result', f.filename)
    recognize_gj(os.path.join(basepath, 'static\\uploadFiles', f.filename), out_path)
    # out_path = os.path.join(basepath, 'static\\uploadFiles\\final_result', f.filename)  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
    # out_path =
    # f.save(upload_path, out_path)
    return "http://127.0.0.1:8080\\static\\uploadFiles\\final_result\\" + f.filename

    # import base64
    # img_stream = ''
    # with open(os.path.join(basepath, 'static\\uploadFiles', f.filename), 'r') as img_f:
    #     img_stream = img_f.read()
    #     img_stream = base64.b64encode(img_stream)
    #     return img_stream

if __name__ == '__main__':
    # 本地测试时可用127.0.0.1，要对外开放时改为0.0.0.0
    app.run(host='127.0.0.1', port=8080,debug=True)
