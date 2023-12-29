from django.http import HttpResponse
from django.shortcuts import render

from VR_APP.service.video_service import VideoService
from VR_Backend.settings import config, video_db, UPLOAD_PATH, BASE_DIR
import json
import os
from django.http import HttpResponse, Http404, FileResponse

video_service = VideoService()

# text to video请求处理
def caption_query(request):
    return render(request, 'VR_APP/caption_query.html')

# video to video 请求处理+视图显示
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

                # video_service = VideoService()
                res_data = video_service.video2video_query(file_name, video_db)
                recommend = {"recommend_list": res_data["video_list"]}
                # return HttpResponse(json.dumps(recommend, indent=4, ensure_ascii=False))
                return render(request, 'VR_APP/result.html', recommend)
            else:
                return HttpResponse("上传文件格式错误")
    else:
        return HttpResponse(render(request, 'VR_APP/upload.html'))

# 视图显示
def query(request):
    if request.method == 'POST':
        # request的POST方法获取表单的数据对象
        data = request.POST
        caption = data.get('caption')
        # 验证caption是否为空
        if caption != None and caption.strip() != "":
            # 构建视频服务
            # video_service = VideoService()
            res_data = video_service.text2video_query(caption, video_db)
        recommend = {"recommend_list": res_data["video_list"]}
        print(recommend)
        # return HttpResponse(json.dumps(recommend, indent=4, ensure_ascii=False))
        return render(request, 'VR_APP/result.html', recommend)


def file_download(request, file_name):
    try:
        print(file_name)
        file_path = "/root/dataset/data/MSRVTT_Videos/" + file_name + ".mp4"
        response = FileResponse(open(file_path, 'rb'))
        response['content_type'] = "application/octet-stream"
        response['Content-Disposition'] = 'attachment; filename=' + file_name + '.mp4'
        return response
    except Exception:
        raise Http404


def test_caption_query(request):
    if request.method == "POST":
        request_json = json.loads(request.body)
        print(request_json)
        caption = request_json.get("caption")
        if caption != None and caption.strip() != "":
            rank = request_json.get("rank")
            res_data = video_service.text2video_query(caption, video_db, rank)
            path_list = ["/root/dataset/data/MSRVTT_Videos" + x + ".mp4" for x in res_data["video_list"]]
            res_data["path_list"] = path_list
            response_json = json.dumps(res_data, indent=4)
            return HttpResponse(response_json)
    
    return HttpResponse("error")
    