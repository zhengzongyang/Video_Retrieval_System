from django.http import HttpResponse
from django.shortcuts import render

from VR_APP.service.video_service import VideoService
from VR_Backend.settings import config, video_db, UPLOAD_PATH
import json
import os

def query(request):
    if request.method == 'POST':
        request_str = request.body
        request_json = json.loads(request_str)
        print('request infomation:',request_json)
        caption = request_json["caption"]
        # 验证caption是否为空
        if caption != None and caption.strip() != "":
            # 构建视频服务
            video_service = VideoService()
            recommend_list = video_service.text2video_query(caption, video_db)
        recommend = {"recommend_list": recommend_list}
        return HttpResponse(json.dumps(recommend, indent=4, ensure_ascii=False))


def upload_file(request):
    if request.method == "POST":
        print("进入上传文件视图")
        file = request.FILES.get('file1')
        if file is None:
            return HttpResponse("no files upload")
        else:
            # 获取项目名
            file_name = file.name
            # 判断是否为mp4文件
            if file_name.endswith(".mp4"):
                # 判断是否存在上传的文件夹

                video_save_path = os.path.join(UPLOAD_PATH,"videos")
                video_path = os.path.join(video_save_path,file_name)
                if not os.path.exists(video_save_path):
                    os.makedirs(video_save_path)

               # 写入上传文件
                with open(video_path, "wb+") as f:
                    for chunk in file.chunks():
                        f.write(chunk)

                video_service = VideoService()
                recommend_list = video_service.video2video_query(file_name, video_db)
                recommend = {"recommend_list": recommend_list}
                return HttpResponse(json.dumps(recommend, indent=4, ensure_ascii=False))
            else:
                return HttpResponse("上传文件格式错误")
    else:
        return HttpResponse(render(request, 'VR_APP/upload.html'))