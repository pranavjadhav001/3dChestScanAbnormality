from PIL import Image
import numpy as np
import glob
import cv2
import scipy.ndimage
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
import os

def image_label():
    files = glob.glob(r'F:/notes/sample_dataset_for_testing/fullsampledata/subset1mask/1.3.6.1.4.1.14519.5.2.1.6279.6001.108197895896446896160048741492/*.tiff')
    image = []
    for i in files:
        if i.split('_')[-1] != 'mask.tiff':
            img = np.array(Image.open(i))
            img = cv2.resize(img,(200,200))
            image.append(img)
    image = np.array(image)
    return image
def alloperations(pathfile):
    def coordinates(a):
        list1 = []
        list2 = []
        X=[]
        Y=[]
        Z=[]
        B=a.reshape(a.shape[0]*a.shape[1]*a.shape[2],1)
        print(B.shape)
        C=list(itertools.product(range(a.shape[0]),range(a.shape[1]),range(a.shape[2])))
        for i,n in enumerate(B):
            if n != 0:
                #print(n)
                list1.append(i)
        for j in list1:
            list2.append(C[j])
        for j in range(len(list2)):
            X.append(list2[j][0])
            Y.append(list2[j][1])
            Z.append(list2[j][2])
        return X,Y,Z

    def image_label(path):
        files = glob.glob(os.path.join(path,'*.tiff'))
        image = []
        for i in files:
            if i.split('_')[-1] != 'mask.tiff':
                img = np.array(Image.open(i))
                img = cv2.resize(img,(200,200))
                image.append(img)
        image = np.array(image)
        return image


    def make_label(path,image):
        mask_files= glob.glob(os.path.join(path,'*_mask.tiff'))
        dicti = {}
        for i in mask_files:
            dicti[i.split('\\')[-1].split('_')[0]] = i
        labels = []
        #print(len(files))
        for i in range(image.shape[0]):
            if str(i) in dicti:
                img = Image.open(dicti[str(i)])
                labels.append(cv2.resize(np.array(img),(200,200)))
            else:
                img = Image.open(r'F:/notes/sample_dataset_for_testing/fullsampledata/subset2mask/1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059/0.tiff')
                labels.append(cv2.resize(np.array(img),(200,200)))
        labels = np.array(labels)
        return labels          

    image = image_label(pathfile)
    label = make_label(pathfile,image)

    def plot_3d(image,label,threshold=500):
        if image.shape == label.shape:
        # Position the scan upright, 
        # so the head of the patient would be at the top facing the camera
            p = image.transpose(2,1,0)
            label = label.transpose(2,1,0)
            xdata,ydata,zdata =coordinates(label)   
            verts, faces,normal,values = measure.marching_cubes_lewiner(p,threshold)

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')

            # Fancy indexing: `verts[faces]` to generate a collection of triangles
            mesh = Poly3DCollection(verts[faces], alpha=0.70)
            face_color = [0.15, 0.15, 0.45]
            mesh.set_facecolor(face_color)
            ax.add_collection3d(mesh)
            ax.scatter(xdata, ydata, zdata,c='r',marker='o')
            ax.set_xlim(0, p.shape[0])
            ax.set_ylim(0, p.shape[1])
            ax.set_zlim(0, p.shape[2])

            plt.show ()
        else:
            print('pls chck')

    #plot_3d(image,label)
    return image,label

  

def image_show(image):
    fig = plt.figure()
    for num,each_slice in enumerate(image):
        y = fig.add_subplot(31,10,num+1)
        y.imshow(each_slice)
    image_show(image)
    plt.show()

def chunks(l, n):
    # Credit: Ned Batchelder
    # Link: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def mean(l):
    return sum(l) / len(l)
##image= image_label()
##HM_SLICES = 20
##chunk_sizes = math.ceil(len(image)/HM_SLICES)
##print(chunk_sizes)
##new_slices = []
##
##
##for slice_chunk in chunks(image,chunk_sizes):
##        slice_chunk = list(map(mean, zip(*slice_chunk)))
##        new_slices.append(slice_chunk)
##new_slices = np.array(new_slices)        

#plot_3d(image)
def directory_name():
    path = r'F:/notes/sample_dataset_for_testing/fullsampledata'
    paths = []
    for i in os.listdir(path):
        new_path = os.path.join(path,i)
        
        for j in os.listdir(new_path):
            final_path = os.path.join(new_path,j)
            paths.append(final_path)
        
    return paths
def make_chunks(image):
    HM_SLICES = 20
    chunk_sizes = math.ceil(len(image)/HM_SLICES)
    print(chunk_sizes)
    new_slices = []
    for slice_chunk in chunks(image,chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(np.array(slice_chunk))
    
    if len(new_slices) == HM_SLICES-1:
        new_slices.append(new_slices[-1])

    if len(new_slices) == HM_SLICES-2:
        new_slices.append(new_slices[-1])
        new_slices.append(new_slices[-1])

    if len(new_slices) == HM_SLICES+2:
        new_val = list(map(mean, zip(*[new_slices[HM_SLICES-1],new_slices[HM_SLICES],])))
        del new_slices[HM_SLICES]
        new_slices[HM_SLICES-1] = new_val
        
    if len(new_slices) == HM_SLICES+1:
        new_val = list(map(mean, zip(*[new_slices[HM_SLICES-1],new_slices[HM_SLICES],])))
        del new_slices[HM_SLICES]
        new_slices[HM_SLICES-1] = new_val

    new_slices = np.array(new_slices)
    return new_slices

def final_labels_images():
    paths = directory_name()
    data = []
    label_data = []
    for i in range(len(paths)):
        image,label = alloperations(paths[i])
        image = make_chunks(image)
        label = make_chunks(label)
        print(image.shape,label.shape)
        data.append(image)
        label_data.append(label)
    return  data,label_data

