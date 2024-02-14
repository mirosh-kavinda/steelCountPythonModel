import os
import numpy as np
import tensorflow as tf
import math
import cv2
from flask import Flask, request, jsonify, send_file
from skimage import transform

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

graph = tf.Graph()
with graph.as_default():
    sess = tf.compat.v1.Session()
    with sess.as_default():
        saver = tf.compat.v1.train.import_meta_graph('model/model.ckpt.meta')
        saver.restore(sess, tf.compat.v1.train.latest_checkpoint('model/'))

def detection(img_original,stride):
    height = np.size(img_original,0)
    width = np.size(img_original,1)
    img_original = transform.resize(img_original,(height,width))   
    patch_size = 71
    imgs=[]
    coordidate = []
    for i in range(patch_size,height-patch_size,stride):
        for j in range(patch_size,width-patch_size,stride):
            img_original_patch = img_original[int(i-(patch_size-1)/2):int(i+(patch_size-1)/2+1),int(j-(patch_size-1)/2):int(j+(patch_size-1)/2+1),:]
            imgs.append(img_original_patch)
            coordidate.append([i,j])
    data = np.asarray(imgs,np.float32)
    output =[]
    x = graph.get_tensor_by_name("x:0")
    vol_slice = 5000
    num_slice = math.ceil(np.size(data,0)/vol_slice)
    for i in range(0,num_slice,1):
        if i+1 != num_slice:
            data_temp = data[i*vol_slice:(i+1)*vol_slice]            
        else:
            data_temp = data[i*vol_slice:np.size(data,0)]
            
        feed_dict = {x:data_temp}
        logits = graph.get_tensor_by_name("logits_eval:0")
        classification_result = sess.run(logits,feed_dict)
        output_temp = sess.run(tf.argmax(classification_result, 1), feed_dict=feed_dict)
        output = np.hstack((output,output_temp))   
    candidate_center = []
    for i in range(len(output)):
        if output[i] == 1:
            candidate_center.append(coordidate[i])            
    return np.array(candidate_center) #the matrix of candidate center

# the function of clustering, which gets the group of candidate center
def clustering(candidate_center,threshold_dis):
    x = candidate_center[:,1]
    y = candidate_center[:,0]
    group_distance = []
    for i in range(len(candidate_center)):
        xpoint, ypoint = x[i], y[i]
        xTemp, yTemp = x, y 
        distance = np.sqrt(pow((xpoint-xTemp),2)+pow((ypoint-yTemp),2))
        distance_matrix = np.vstack((np.array(range(len(candidate_center))),distance))
        distance_matrix = np.transpose(distance_matrix)
        distance_sort = distance_matrix[distance_matrix[:,1].argsort()] 
        distance_sort = np.delete(distance_sort,0,axis = 0)
        thre_matrix = distance_sort[distance_sort[:,1]<=threshold_dis]
        thre_point = thre_matrix[:,0]
        thre_point = thre_point.astype(int)
        thre_point = thre_point.tolist()
        thre_point.insert(0,i)
        group_distance.append(thre_point)
    
    group_clustering = [[]] 
    
    for i in range(len(candidate_center)):
        m1 = group_distance[i]
        for j in range(len(group_clustering)):
            m2 = group_clustering[j]
            com = set(m2).intersection(set(m1))
            if len(com) == 0:
                if j == len(group_clustering)-1:
                    group_clustering.append(m1)
            else:
                m = set(m1).union(set(m2))
                group_clustering[j] = []
                group_clustering[j] = list(m)
                break
    group_clustering.pop(0)
    return group_clustering  #the group of candiate center

#the function of clustering the final center
def center_clustering(candidate_center,group_clustering):
    final_result = []
    for i in range(len(group_clustering)): 
        points_coord = candidate_center[group_clustering[i]]
        xz = points_coord[:,1] 
        yz = points_coord[:,0]
        x_mean = np.mean(xz)
        y_mean = np.mean(yz)
        final_result.append([y_mean,x_mean])
    final_result = np.array(final_result)
    final_result = final_result.astype(np.int32)
    return final_result 

def show_clustering_red_point(img_original,center_cluster,output_file):  
    for i in range(len(center_cluster)):
        cv2.circle(img_original,(center_cluster[i,1],center_cluster[i,0]),2,(0,0,255),-1) 
    cv2.imwrite (output_file, img_original)
    print("Output Image saved.")

app = Flask(__name__)
# CORS(app) 
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    req_image = request.files['image']

    # Save the image temporarily
    temp_path = 'temp_image.jpg'
    output_path = 'output_image.jpg'
    req_image.save(temp_path)
# Get stride and distance threshold from request parameters
    stride = 6
    distance_threshold = 18


    # Run Model 
    with graph.as_default():
        img_original = cv2.imread(temp_path)
        candidate_center = detection(img_original,stride)   
        group_clustering = clustering(candidate_center,distance_threshold)       
        center_cluster = center_clustering(candidate_center,group_clustering)
            
    steal_count=len(center_cluster);


    show_clustering_red_point(img_original,center_cluster,output_path)

    return jsonify({
        "stealcount": f'{steal_count}',
        "outputimage_url": f"/get_image/{output_path}"
    })

@app.route('/get_image/<filename>', methods=['GET'])
def get_image(filename):
    # Send the saved image file
    return send_file(filename, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
