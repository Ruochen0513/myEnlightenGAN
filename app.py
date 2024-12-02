from flask import Flask, request, render_template, send_file
import os
import subprocess
import shutil
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 配置上传文件夹和结果文件夹
UPLOAD_FOLDER = './use_decom_dataset/test'
RESULT_FOLDER = './use_decom_dataset/result'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    clear_folder(UPLOAD_FOLDER)
    clear_folder(RESULT_FOLDER)
    clear_folder('./ablation/enlightening/test_200/images')
    clear_folder('./use_decom_dataset/cache')
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
            result_filename = f'result_{base_filename}.png'
            
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
            result_path = os.path.join(RESULT_FOLDER, result_filename)
            if os.path.exists(result_path):
                return render_template('result.html', original=filename, result=result_filename)
            else:
                return '处理失败'
            
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/results/<filename>')
def result_file(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename))

if __name__ == '__main__':
    app.run(debug=True)