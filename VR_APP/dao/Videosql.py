import json
import pymysql
import re
import numpy as np


class Videosql():
    def __init__(self, config) -> None:
        self.database_config = {'host':config['host'], 'port':int(config['port']), 'user':config['user'], 'password':config['password'], 'database':config['database']}
        self.datatable = config['datatable']
        self.feature_length = 512

    def __enter__(self): 
        try:
            self.conn = pymysql.connect(
                            host=self.database_config['host'],
                            port=self.database_config['port'],
                            user=self.database_config['user'],
                            password=self.database_config['password'],
                            database=self.database_config['database'])
            self.cursor = self.conn.cursor()

            #判断数据表是否存在，不存在则创建
            sql = "show tables;"
            self.cursor.execute(sql)
            tables = [self.cursor.fetchall()]
            # TODO:这两句可以优化一下，就是判断数据库中有没有这张表
            table_list = re.findall('(\'.*?\')',str(tables))
            table_list = [re.sub("'",'',each) for each in table_list]
            if self.datatable not in table_list:
                self.creat_datatable(self.datatable)
        except Exception as e:
            raise Exception('Fail to connect db! {}'.format(str(e)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('Video Database closed!')
        self.cursor.close()
        self.conn.close()

    def creat_datatable(self, datatable):
        sql = """
            create table IF NOT EXISTS `%s`(
                video_id int primary key AUTO_INCREMENT,
                video_name varchar(255),
                video_num int,
                scene_time varchar(255),
                fps int,
                video_feat blob,
                caption varchar(1024),
                file_path varchar(1024)
            )
        """%(datatable)
        self.cursor.execute(sql)
        self.conn.commit()

    def get_all_feature(self):                                  #从mysql中查询所有的人脸特征                                  
        sql = "SELECT `video_feat` FROM %s" %(self.datatable)
        self.cursor.execute(sql)
        resstrs=self.cursor.fetchall()
        resarray = np.zeros([len(resstrs), self.feature_length], dtype=np.float32)
        # print(np.shape(resarray))
        # print(type(eval(resstrs[0][0])))
        for idx, resstr in enumerate(resstrs):
            # 我这边不能直接将列表转换为Numpy Array
            resarray[idx,:] = np.array(eval(resstr[0]), dtype = np.float32)
        return np.array(resarray, dtype = np.float32)
    
    def query_list_data(self, data_list):
        sql = "SELECT `video_name` FROM {table} WHERE video_id in ({seq})".format(table=self.datatable,
            seq=','.join([r'%s'] * len(data_list)))
        self.cursor.execute(sql, data_list)
        resstr = self.cursor.fetchall()
        recommend_list = re.findall('\'(.*?)\'',str(resstr))
        print(recommend_list)
        return recommend_list

        

        



if __name__ == '__main__':
    with open(r'/root/workspace_qlab/VR_Backend/VR_APP/config/DatabaseConfig.json','r',encoding='utf8') as fp:
        config = json.load(fp)
    fp.close()
    print(config['videobase'])
    with Videosql(config['videobase']) as videosql:
        video_db = videosql.get_all_feature()
        print(video_db.dtype)
        # index_list = [1,2,3]
        # videosql.query_list_data(index_list)
        print("初始化数据库配置")