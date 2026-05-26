import cv2
import numpy as np


def water_mark(img,mask,alpha=0.25):
    assert (img.shape[0]==mask.shape[0]) and (img.shape[1]==mask.shape[1]), "Wrong size"
    if len(img.shape)<3:
        flag=True
        t_img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGBA)
    else:
        flag=False
        t_img=cv2.cvtColor(img,cv2.COLOR_RGB2RGBA)
    if (mask.dtype==bool):
        t_mask=cv2.cvtColor((mask*255).astype(np.uint8),cv2.COLOR_GRAY2RGBA)
    elif (mask.dtype==np.uint8):
        if len(mask.shape)<3:
            t_mask=cv2.cvtColor((mask).astype(np.uint8),cv2.COLOR_GRAY2RGBA)
        else:
            t_mask=cv2.cvtColor((mask).astype(np.uint8),cv2.COLOR_RGB2RGBA)
    else:
        if len(mask.shape)<3:
            t_mask=cv2.cvtColor((mask*255).astype(np.uint8),cv2.COLOR_GRAY2RGBA)
        else:
            t_mask=cv2.cvtColor((mask*255).astype(np.uint8),cv2.COLOR_RGB2RGBA)
    t_out=cv2.addWeighted(t_img,1,t_mask,alpha,0)
    if flag:
        out=cv2.cvtColor(t_out,cv2.COLOR_RGBA2GRAY)
    else:
        out=cv2.cvtColor(t_out,cv2.COLOR_RGBA2RGB)
    return out

def put_data(img,data,binary_mask=np.uint8(1)):
    assert img.dtype==np.uint8 , "img wrong data type"
    assert binary_mask.dtype==np.uint8, "binary_mask wrong data type"
    un_binary_mask=np.unpackbits(binary_mask)
    if data.dtype!=bool:
        unpacked_data=np.unpackbits(data)
    else:
        unpacked_data=data
    dataspace=img.shape[0]*img.shape[1]*np.sum(un_binary_mask)
    assert (dataspace>=unpacked_data.size) , "too much data"
    if dataspace==unpacked_data.size:
        prepered_data=unpacked_data.reshape(img.shape[0],img.shape[1],np.sum(un_binary_mask)).astype(np.uint8)
    else:
        prepered_data=np.resize(unpacked_data,(img.shape[0],img.shape[1],np.sum(un_binary_mask))).astype(np.uint8)
    mask=np.full((img.shape[0],img.shape[1]),binary_mask)
    img=np.bitwise_and(img,np.invert(mask))
    bv=0
    for i,b in enumerate(un_binary_mask[::-1]):
        if b:
            temp=prepered_data[:,:,bv]
            temp=np.left_shift(temp,i)
            img=np.bitwise_or(img,temp)
            bv+=1
    return img

def pop_data(img,binary_mask=np.uint8(1),out_shape=None):
    un_binary_mask=np.unpackbits(binary_mask)
    data=np.zeros((img.shape[0],img.shape[1],np.sum(un_binary_mask))).astype(np.uint8)
    bv=0
    for i,b in enumerate(un_binary_mask[::-1]):
        if b:
            mask=np.full((img.shape[0],img.shape[1]),2**i)
            temp=np.bitwise_and(img,mask)
            data[:,:,bv]=temp[:,:].astype(np.uint8)
            bv+=1
    if out_shape!=None:
        tmp=np.packbits(data.flatten())
        tmp=tmp[:np.prod(out_shape)]
        data=tmp.reshape(out_shape)
    return data

if __name__=="__main__":
    ...
