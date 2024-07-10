import os

USE_NEPTUNE = True
if USE_NEPTUNE:
    import neptune


def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def get_model_save_path():
    if not 'params' in os.listdir('.'):
        os.mkdir('./params/')
        
    if os.listdir('params'):
        last_save_id = max([int(file_name.split('_')[1]) for file_name in os.listdir('params')])
    else:
        last_save_id = 0
    new_save_id = last_save_id + 1
    os.mkdir(f'params/params_{new_save_id}')
    model_save_path = f'params/params_{new_save_id}'
    return model_save_path

class CustomList(list):
    def upload_files(self, *args):
        pass
    
    def upload(self, *args):
        pass

class NeptuneToy:
    def __init__(self):
        self.toy_list = CustomList()

    def __getitem__(self, key):
        return self.toy_list
    
    def stop(self, ):
        return
        

def init_neptune(tags=[], mode='async'):
    
    
    if USE_NEPTUNE:
        run = neptune.init_run(
            
            project="nazimbendib1/SLL-strength",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MGIxMzhlZS00MzhhLTQ0ZDktYTU2Yy0yZDk3MjE4MmU4MDgifQ==",
            
            # project="nazim-bendib/simclr-rl",
            # api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNDVjNWJkYi1mMTIwLTRmNDItODk3Mi03NTZiNzIzZGNhYzMifQ==",
            
            # project="nazimbendib1/SIMCLR",
            # api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MGIxMzhlZS00MzhhLTQ0ZDktYTU2Yy0yZDk3MjE4MmU4MDgifQ==",
            
            tags=tags,
            mode=mode
        )
    else:
        run = NeptuneToy()
    
    return run