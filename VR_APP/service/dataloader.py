import numpy as np

from VR_APP.service.utils.rawframe_util import RawFrameExtractor

class Text_DataLoader():
    def __init__(
            self,
            tokenizer,
            max_words=32,
    ):
        self.max_words = max_words
        self.tokenizer = tokenizer

        # start and end token
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def get_text(self, caption):
        """
            get tokenized word feature

            Args:
                caption: caption

            Returns:
                pairs_text: tokenized text
                pairs_mask: mask of tokenized text
                pairs_segment: type of tokenized text

        """

        # tokenize word
        words = self.tokenizer.tokenize(caption)

        # add cls token
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]

        # add end token
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

        # convert token to id according to the vocab
        input_ids = self.tokenizer.convert_tokens_to_ids(words)

        # add zeros for feature of the same length
        # 所以mask是为了特征变成等长的
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        # ensure the length of feature to be equal with max words
        assert len(input_ids) == self.max_words
        assert len(input_mask) == self.max_words
        assert len(segment_ids) == self.max_words
        pairs_text = np.array(input_ids)
        pairs_mask = np.array(input_mask)
        pairs_segment = np.array(segment_ids)

        # TODO:segement作用是什么,给语句分段，如果是一句话可能没什么影响
        # TODO:这边初始化能不能优化一下
        pairs_text = np.reshape(pairs_text, (1, 32))
        pairs_mask = np.reshape(pairs_mask, (1, 32))
        pairs_segment = np.reshape(pairs_segment, (1, 32))

        return pairs_text, pairs_mask, pairs_segment
        

class Visual_DataLoader():
    def __init__(
            self,
            feature_framerate=2,
            max_frames=24,
            image_resolution=224,
    ):
        self.feature_framerate = feature_framerate
        self.max_frames = max_frames

        # frame extractor to sample frames from video
        self.frameExtractor = RawFrameExtractor(framerate=feature_framerate, size=image_resolution)
    # video_path = features_path + video_id
    def get_rawvideo(self, video_path):
        """get sampled frames

        Args:
            video_id: id of video

        Returns:
            video: sampled frame
            video_mask: mask of video
        """
        # 1 x 12
        video_mask = np.zeros((1, self.max_frames), dtype=np.int64)

        # 1 x L x 1 x 3 x H x W = 1 x 12 x 1 x 3 x H x w
        video = np.zeros((1, self.max_frames, 1, 3,
                          self.frameExtractor.size, self.frameExtractor.size), dtype=float)

        # get sampling frames 返回的格式已经是tensor形式了
        raw_video_data = self.frameExtractor.get_video_data(video_path, self.max_frames)
        raw_video_data = raw_video_data['video']
        # print(raw_video_data.shape)

        # L x 1 x 3 x H x W
        if len(raw_video_data.shape) > 3:
            raw_video_data_clip = raw_video_data
            # L x T x 3 x H x W
            raw_video_slice = self.frameExtractor.process_raw_data(raw_video_data_clip)
            # max_frames x 1 x 3 x H x W
            if self.max_frames < raw_video_slice.shape[0]:
                    sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                    video_slice = raw_video_slice[sample_indx, ...]
            else:
                video_slice = raw_video_slice

            # set max length, and save video mask and frames
            slice_len = video_slice.shape[0]
            video_mask[0][:slice_len] = [1] * slice_len
            video[0][:slice_len, ...] = video_slice # 难道是自动格式转换

        else:
            print("get raw video error, skip it.")

        return video, video_mask
