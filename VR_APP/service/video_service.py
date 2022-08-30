import numpy as np
import torch
import os
from VR_APP.service.dataloader import Text_DataLoader,Visual_DataLoader
from VR_APP.preload import tokenizer, encoder, args
from VR_Backend.settings import config, UPLOAD_PATH
from VR_APP.dao.Videosql import Videosql
import faiss



class VideoService():
    def __init__(self) -> None:
        self.index = faiss.IndexFlatIP(512)
        with Videosql(config['videobase']) as videosql:
            # 建立索引
            self.index.add(videosql.get_all_feature())
            print("video.index.ntotal:%d" % self.index.ntotal)
        pass

    def get_video(self, video_path, max_frames, frame_rate=2):
        """
        函数功能：根据视频地址得到采样的帧序列和序列掩码
        输入：帧序列地址,最大帧序列长度，帧的采样间隔
        返回：帧序列、帧序列掩码
        """
        visualData = Visual_DataLoader(frame_rate, max_frames)
        video, video_mask = visualData.get_rawvideo(video_path)
        # 增加一个维度
        video = np.reshape(video, (1, 1, max_frames, 1, 3, 224, 224))
        video_mask = np.reshape(video_mask, (1, 1, max_frames))
        
        video_mask = torch.as_tensor(video_mask).to("cuda:0")
        
        return video, video_mask
    
    def get_visualFeature(self, visual_encoder, video, video_mask):
        """
        函数功能：得到视觉编码输出
        输入：视觉编码器、帧序列、帧序列掩码
        输出：视觉特征
        """
        visualFeature = visual_encoder.get_visual_feature(video, video_mask)
        return visualFeature

    def get_overall_video_feature(self, visual_encoder, video_path, max_frames, frame_rate=2):
        """
        函数功能：获取一个视频的全局特征，向量维度：1 x 512
        输入：视觉编码器、视频路径
        输出：视频特征(1 x 512)
        """

        video, video_mask = self.get_video(video_path, max_frames,frame_rate)
        visualFeature = self.get_visualFeature(visual_encoder, video, video_mask)
        video_feature = visual_encoder.get_video_feature(visualFeature, video_mask)
        video_feature = video_feature.cpu().detach().numpy()

        return video_feature

    def get_text(self, tokenizer, caption):
        """
        函数功能：对文本进行预处理
        返回：文本中的单词对应于vocab的id，单词的mask，单词段落标记
        """
        text_dataloader = Text_DataLoader(tokenizer)
        pairs_text, pairs_mask, pairs_segment = text_dataloader.get_text(caption)

        pairs_text = torch.as_tensor(pairs_text).to("cuda:0")
        pairs_mask = torch.as_tensor(pairs_mask).to("cuda:0")
        pairs_segment = torch.as_tensor(pairs_segment).to("cuda:0")

        return pairs_text, pairs_mask, pairs_segment


    def get_overall_text_feature(self, text_encoder, tokenizer, caption):
        """
        函数功能：得到文本的全局特征
        输入：文本编码器、词语分割器、文本
        返回：文本特征(1 x 512)
        """
        pairs_text, pairs_mask, pairs_segment = self.get_text(tokenizer, caption)
        sequence_output = text_encoder.get_sequence_output(pairs_text, pairs_segment, pairs_mask, shaped=True)
        text_feature = text_encoder.get_text_feature(sequence_output, pairs_mask)
        # 将文本特征从GPU->CPU上处理
        text_feat = text_feature.cpu().detach().numpy()

        return text_feat 

    def softmax(self, x):
        """ softmax function """

        # assert(len(x.shape) > 1, "dimension must be larger than 1")
        # print(np.max(x, axis = 1, keepdims = True)) # axis = 1, 行

        x -= np.max(x, axis = 1, keepdims = True) #为了稳定地计算softmax概率， 一般会减掉最大的那个元素
        x = np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)

        return x

    """
    得到的跨模态特征在数据库中查询匹配视频
    """
    def match(self, feature, video_db, rank):
        video_cnt, feat_dim = video_db.shape
        # 与特征库特征矩阵进行点乘计算相似度
        match_result = video_db.dot(feature.T).reshape(1, video_cnt)
        match_result_softmax = self.softmax(match_result)

        # 对得分大小进行排序，得分从小到大
        sortIndex = match_result_softmax[0].argsort()
        # 对得分序列从后向前取5个，则得到的rank个候选索引所对应的得分从大到小
        selectIndex = sortIndex[-1: -rank-1: -1]
        # 对数据库索引进行偏移修正，即索引进行加1操作
        query_index_list = (selectIndex + 1).tolist()

        # 根据查出来的视频索引信息，去数据库中查对应的视频信息
        with Videosql(config['videobase']) as videosql:
            result = videosql.query_list_data(query_index_list)
            
        return result

    def match_faiss(self, feature, rank):
        D, I = self.index.search(feature, rank)
        query_index_list = I[:rank][0] + 1
        print("faiss_index.....")
        # 根据查出来的视频索引信息，去数据库中查对应的视频信息
        with Videosql(config['videobase']) as videosql:
            result = videosql.query_list_data(query_index_list)
        
        return result
        

    def text2video_query(self, caption, video_db, rank=5):
        # 获取文本编码器
        text_encoder = encoder
        text_feat = self.get_overall_text_feature(text_encoder, tokenizer, caption)
        # result = self.match(text_feat, video_db, rank)
        result = self.match_faiss(text_feat, rank)
        return result


    def video2video_query(self, file_name, video_db, rank=5):
        visual_encoder = encoder
        frame_path = os.path.join(UPLOAD_PATH, "samples", file_name.split(".")[0])
        if not os.path.exists(frame_path):
            os.makedirs(frame_path)
        # 将视频处理成帧序列
        self.extract_frames(os.path.join(UPLOAD_PATH, "videos", file_name), frame_path)
        # 将帧序列送入编码器得到特征编码
        video_feature = self.get_overall_video_feature(visual_encoder, frame_path, args.max_frames)
        result = self.match(video_feature, video_db, rank)
        return result


    def extract_frames(self, video_name, out_folder, fps=5):
        """
        将视频处理成帧序列
        """
        if os.path.exists(out_folder):
            os.system('rm -rf ' + out_folder + '/*')
            os.system('rm -rf ' + out_folder)
        os.makedirs(out_folder)
        # -v设置日志级别 -i 指定输入文件名 -r 设定视频流的帧率 -q 图像质量
        cmd = 'ffmpeg -v 0 -i %s -r %d -q 0 %s/%s.jpg' % (video_name, fps, out_folder, '%08d')
        os.system(cmd)