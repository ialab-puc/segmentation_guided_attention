## Data loader class
from torch.utils.data import Dataset

class PlacePulseDataset(Dataset):
    
    def __init__(self,csv_file,img_dir,transform=None, cat=None, equal=False):
        self.placepulse_data = pd.read_csv(csv_file)
        if cat:
            self.placepulse_data = self.placepulse_data[self.placepulse_data['category'] == cat]
        if not equal:
            self.placepulse_data = self.placepulse_data[self.placepulse_data['winner'] != 'equal']
        
        self.img_dir =  img_dir
        self.transform = transform
        self.label = {'left':1, 'right':-1,'equal':0}
    
    def __len__(self):
        return len(self.placepulse_data)
    
    def __getitem__(self,idx):
        
        if type(idx) == torch.Tensor:
            idx = idx.tolist()
        left_img_name = os.path.join(self.img_dir, '{}.jpg'.format(self.placepulse_data.iloc[idx, 0]))
        left_image = io.imread(left_img_name)
        right_img_name = os.path.join(self.img_dir, '{}.jpg'.format(self.placepulse_data.iloc[idx, 1]))
        right_image = io.imread(right_img_name)
        winner = self.label[self.placepulse_data.iloc[idx, 2]]
        cat = self.placepulse_data.iloc[idx, -1]
        sample = {'left_image': left_image, 'right_image':right_image,'winner': winner, 'cat':cat}
        if self.transform:
            sample = self.transform(sample)
        return sample

#  Transformers 

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        left_image, right_image = sample['left_image'], sample['right_image']
        
        return {'left_image': ToTensor.transform_image(left_image),
                'right_image': ToTensor.transform_image(right_image),
                'winner': sample['winner'],
                'cat': sample['cat']}
    @classmethod
    def transform_image(cls,image):
        return torch.from_numpy(image.transpose((2, 0, 1))).float()
    
class Rescale():
    
    def __init__ (self,output_size):
        self.output_size = output_size
    
    def __call__(self, sample):
        left_image, right_image = sample['left_image'], sample['right_image']
        
        return {'left_image': transform.resize(left_image,self.output_size,anti_aliasing=True,mode='constant'),
                'right_image': transform.resize(right_image,self.output_size,anti_aliasing=True,mode='constant'),
                'winner': sample['winner'],
                'cat': sample['cat']}
        