from flask import Flask, request, render_template, send_file
import os
import subprocess
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 配置上传文件夹
UPLOAD_FOLDER = './test_dataset/testA'
RESULT_FOLDER = './ablation/enlightening'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 检查是否有文件
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        
        if file and allowed_file(file.filename):
            # 保存上传的文件
            filename = secure_filename(file.filename)
            base_filename = filename.rsplit('.', 1)[0]
            result_filename = f'{base_filename}_fake_B.png'
            
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # 运行预测脚本
            subprocess.run(['python', 'predict.py', 
                            '--dataroot', './test_dataset',
                            '--name', 'enlightening',
                            '--model', 'single',
                            '--which_direction', 'AtoB',
                            '--no_dropout',
                            '--dataset_mode', 'unaligned',
                            '--which_model_netG', 'sid_unet_resize',
                            '--skip', '1',
                            '--use_norm', '1',
                            '--use_wgan', '0',
                            '--self_attention',
                            '--times_residual',
                            '--instance_norm', '0',
                            '--resize_or_crop', 'no',
                            '--which_epoch', '200'])
            
            # 获取处理后的图片路径
            result_path = os.path.join(RESULT_FOLDER, 'test_200', 'images', result_filename)
            if os.path.exists(result_path):
                return render_template('result.html', original=filename,result=result_filename)
            else:
                return '处理失败'
            
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/results/<filename>')
def result_file(filename):
    return send_file(os.path.join(RESULT_FOLDER, 'test_200', 'images', filename))

if __name__ == '__main__':
    app.run(debug=True) 