import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
import statistics as st
image1 = cv2.imread("depth_files/chess/im0.png")
image2 = cv2.imread("depth_files/chess/im1.png")
image1_copy = image1.copy()
image2_copy = image2.copy()
#Resize images due to computational limitations
image1_s = cv2.resize(image1,(0,0),fx = 0.5,fy=0.5, interpolation = cv2.INTER_AREA)
image2_s = cv2.resize(image2,(0,0),fx = 0.5,fy=0.5, interpolation = cv2.INTER_AREA)
image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
image1_copy = image1.copy()
image2_copy = image2.copy()
#Intrinsic Matrices k_l and k_r
k_l = np.array([[1758.23 ,0, 829.5],[0, 1758.23, 552.78],[0,0,1]])
k_r = np.array([[1758.23 ,0, 829.5],[0, 1758.23, 552.78],[0,0,1]])
#image shapes 
height1,width1,n = image1.shape
height2,width2,n = image2.shape

def calculate_disparity(image1, image2):
    image1 = cv2.resize(image1,(0,0),fx = 0.3,fy=0.3, interpolation = cv2.INTER_AREA)
    image2 = cv2.resize(image2,(0,0),fx = 0.3,fy=0.3, interpolation = cv2.INTER_AREA)
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    disparity_map = np.zeros((image2.shape[0], image2.shape[1]), dtype=np.float64)
    for i in range(5,image2.shape[1]-5):
        for j in range(5, image2.shape[0]-5):
            #A window size of 10 pixels is selected in image 2
            w1 = image2[j-5:j+5,i-5:i+5]
            SSD_max = np.inf
            corrosponding_index = 0
            for k in range(i-50, i+50, 4):
                if (k-5) < 0 or (k+5) > image1.shape[1]-1:
                    continue
                #A window size of 10 pixels is selected in image 1
                w2 = image1[j-5:j+5,k-5:k+5]
                #Performing sum of squared differences between the window pixels
                diff = w1 - w2
                difference = np.array(diff)
                sq = np.square(diff)
                SSD = sq.sum()
                if SSD < SSD_max:
                    SSD_max = SSD
                    corrosponding_index = k      
            disparity_map[j][i] = i - corrosponding_index
    disparity_map = disparity_map + np.abs(np.min(disparity_map)) + 1
    disparity_map = (disparity_map/np.max(disparity_map))*255
    disparity_map = disparity_map.astype(np.uint8)
    return disparity_map     
def draw_epilines(img1,img2,l_points,r_points,lines):
    w, h,_ = img2.shape
    for line, p1, p2 in zip(lines, l_points, r_points):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -line[2]/line[1]])
        x1, y1 = map(int, [h, -(line[2]+line[0]*h)/line[1]])
        p1_x,p1_y = abs(int(p1[0])),abs(int(p1[1]))
        p2_x,p2_y = abs(int(p2[0])),abs(int(p2[1]))
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, (p1_x,p1_y), 5, color, -1)
        img2 = cv2.circle(img2, (p2_x,p2_y), 5, color, -1)

    return img1, img2
    
def get_rectified_new_points(H1,H2,points_left,points_right):
    #Get the points in right image when applied homography with points in first image and vice versa
    new_points1 = np.zeros((points_left.shape),dtype = np.float32)
    new_points2 = np.zeros((points_left.shape),dtype = np.float32)
    for i in range(points_left.shape[0]):
        p1 = np.array([points_left[i][0],points_left[i][1],1])
        p2 = np.array([points_right[i][0],points_right[i][1],1])
        new_point1 = np.dot(H1,p1)
        new_points1[i][0] = new_point1[0]/new_point1[2]
        new_points1[i][1] = new_point1[1]/new_point1[2]
        new_point2 = np.dot(H2,p2)
        new_points2[i][0] = new_point2[0]/new_point2[2]
        new_points2[i][1] = new_point2[1]/new_point2[2]
    return new_points1,new_points2
def getHomographyMatrices(points_left, points_right, F,w1,h1,left_matches_x,left_matches_y,right_matches_x,right_matches_y):
    finish = True
    while (finish != False):
        try:
            _,Ho1,Ho2 = cv2.stereoRectifyUncalibrated(points_left, points_right, F, imgSize=(w1, h1))
            finish = False
        except:
            F,_,_ = ransac(left_matches_x,left_matches_y,right_matches_x,right_matches_y)
            pass
    return Ho1,Ho2
def disambiguate_camera_pose(R_set,t_set,x_set):
    max_positive = 0
    max_index = 0
    for i in range(len(R_set)):
        positive_points = 0
        R = R_set[i]
        r3 = R[-1].reshape(1,-1)
        t = t_set[i].reshape(-1,1)
        for j in x_set[i]:
            x = j.reshape(-1,1)
            if(r3@(x-t)) > 0 and x[2] > 0:
                positive_points +=1 
        if positive_points > max_positive:
                max_positive = positive_points
                max_index = i
    R,t = R_set[max_index],t_set[max_index]
    return R,t
        
        
def projection_matrix(R1,T1,k):
    Rt = np.hstack((R1,-np.dot(R1,T1.reshape(3,1))))
    proj_matrix = np.dot(k,Rt)
    return proj_matrix
def linear_triangulate(R_set,t_set,k_l,k_r,points_left,points_right):
    x_set=[]
    for i in range(len(R_set)):
        R1,R2 = np.identity(3),R_set[i]
        T1,T2 = np.zeros((3,1)),t_set[i]
        #Projection matrix 1
        p1 = projection_matrix(R1,T1,k_l)
        p1_1,p1_2,p1_3 = p1
        #Projection matrix 2
        p2 = projection_matrix(R2,T2,k_r)
        p2_1,p2_2,p2_3 = p2
        x_i = []
        for left,right in zip(points_left,points_right):
            x1,y1 = left
            x2,y2 = right
            A = [y1*p1_3 - p1_2,p1_1 - x1*p1_3,y2*p2_3 - p2_2,p2_1 - x2*p2_3]
            A = np.array(A).reshape(4,4)
            U,S,Vt = np.linalg.svd(A)
            X = Vt[-1]
            X = X/X[-1]
            x_i.append(X[:-1])
        x_set.append(x_i)
    return x_set

def decompose_essential(E):
    W =  np.array([[0,-1,0],[1,0,0],[0,0,1]])
    R = []
    t = []
    U,S,Vt = np.linalg.svd(E)
    for i in range(1,5):
        if(i==3 or i==4):
            r_i = np.dot(U,np.dot(W.T,Vt))
        elif(i==1 or i==2):
            r_i = np.dot(U,np.dot(W,Vt))
        if(i%2 == 0):
            t_i = -U[:,2]
        else:
            t_i = U[:,2]
        if(round(np.linalg.det(r_i)) == -1):
            r_i = -r_i
            t_i = -t_i
        R.append(r_i)
        t.append(t_i)
    return R,t

def compute_essential_matrix(F,k_l,k_r):
    E = np.dot(k_r.T,np.dot(F,k_l))
    U,S,V = np.linalg.svd(E)
    #Reconstruction of S with singular values
    S = np.diag(S)
    S[0,0] = 1
    S[1,1] = 1
    S[2,2] = 0
    E = np.matmul(U,np.matmul(S,V))
    print("The Essential Matrix for Chess Dataset is:")
    print(E)
    return E
def compute_fundamental_matrix(x1,y1,x2,y2):
    A=[]
    for i in range(len(x1)):
        row1 = [x1[i]*x2[i],x1[i]*y2[i],x1[i],y1[i]*x2[i],y1[i]*y2[i],y1[i],x2[i],y2[i],1]
        A.append(row1)
    A= np.array(A)
    U, s, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)
    U,S,V = np.linalg.svd(F)
    S = np.diag(S)
    S[2,2] = 0
    F = np.dot(U,np.dot(S,V))
    return F

def compute_error(F,x1i,y1i,x2i,y2i):   
    # pt1 = np.array([x1i,y1i,1])
    # pt2 = np.array([x2i,y2i,1])
    pt1 = np.matrix([x1i,y1i,1])
    pt2 = np.matrix([x2i,y2i,1])
    error = np.dot(pt2,np.dot(F,pt1.T))
    # error = np.dot(pt2.T,np.dot(F,pt1))       
    return abs(error)

    
def sift_matcher(img1,img2,image1_color,image2_color):
    img1 = cv2.GaussianBlur(img1, (11, 11),cv2.BORDER_DEFAULT)
    img2 = cv2.GaussianBlur(img2, (11, 11),cv2.BORDER_DEFAULT)
    # Create SIFT Object
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for match1,match2 in matches:
        if match1.distance < 0.35*match2.distance:
            good.append([match1])
    #Extracting all the points values derived from keypoint descriptors
    key_points_1 = np.float32([kp.pt for kp in kp1])
    key_points_2 = np.float32([kp.pt for kp in kp2])
    if len(good) > 8:
        # construct the two sets of points i.e get two points arrays from both the keypoint values which belong to the match array descriptors. So, getting only keypoints
        #in length to matches array
        pointsA = np.array(np.float32([key_points_1[m[0].queryIdx] for m in good]))
        pointsB = np.array(np.float32([key_points_2[m[0].trainIdx] for m in good]))
    
    # F, mask = cv2.findFundamentalMat(np.int32(pointsA),np.int32(pointsB),cv2.FM_LMEDS)
    # print("Fundaemneatkl",F)
    pointsA_x = pointsA[:,0]
    pointsA_y = pointsA[:,1]
    pointsB_x = pointsB[:,0]
    pointsB_y = pointsB[:,1]
    # cv2.drawMatchesKnn expects list of lists as matches.
    sift_matches = cv2.drawMatchesKnn(image1_color,kp1,image2_color,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # print(pointsA_x,len(pointsA_x))
    # print(pointsB_x,len(pointsA_x))
    cv2.imwrite("chess/SIFTMatchingPoints.jpg",sift_matches)
    return (sift_matches,pointsA_x,pointsA_y,pointsB_x,pointsB_y)
def ransac(left_matches_x,left_matches_y,right_matches_x,right_matches_y):
    #Getting 8 random points 
    max_inliers = 0             
    F_final = []
    threshold = 0.08
    iterations = 500
    best_left_points = np.zeros((16)).reshape(-1,2)
    best_right_points = np.zeros((16)).reshape(-1,2)
    for i in range(iterations):
        current_inliers = 0
        random_indices = np.random.choice(len(left_matches_x),size =8,replace=False)
        pa_x_rand = left_matches_x[random_indices]
        pa_y_rand = left_matches_y[random_indices]
        pb_x_rand = right_matches_x[random_indices]
        pb_y_rand = right_matches_y[random_indices]
        #computing Fundamental matrix
        F = compute_fundamental_matrix(pa_x_rand,pa_y_rand,pb_x_rand,pb_y_rand)
        for i in range(len(left_matches_x)):
            error = compute_error(F,left_matches_x[i],left_matches_y[i],right_matches_x[i],right_matches_y[i])
            if(error<threshold):
                current_inliers += 1
        if(current_inliers > max_inliers):
            max_inliers = current_inliers
            F_final = copy.deepcopy(F)
            best_left_points = np.vstack(((pa_x_rand),(pa_y_rand))).T
            best_right_points = np.vstack(((pb_x_rand),(pb_y_rand))).T
    
   
    return F_final,best_left_points,best_right_points

fig = plt.figure(figsize=(10,12))
sift_matches,left_matches_x,left_matches_y,right_matches_x,right_matches_y = sift_matcher(image1_gray,image2_gray,image1_rgb,image2_rgb)
F,best_left_points,best_right_points =  ransac(left_matches_x,left_matches_y,right_matches_x,right_matches_y)
print("The Fundamental Matrix for chess dataset is :")
print(F)
E = compute_essential_matrix(F,k_l,k_r)
R_set,t_set = decompose_essential(E)
points_left = np.vstack(((left_matches_x),(left_matches_y))).T
points_right =  np.vstack(((right_matches_x),(right_matches_y))).T
x_set = linear_triangulate(R_set,t_set,k_l,k_r,points_left,points_right)
R_final , T_final = disambiguate_camera_pose(R_set,t_set,x_set)
print("Rotation matrix for chess dataset is:")
print(R_final)
print("Translation matrix for chess dataset is:")
print(T_final)
#Calculating Homography matrices
H1,H2 = getHomographyMatrices(points_left, points_right, F,width1,height1,left_matches_x,left_matches_y,right_matches_x,right_matches_y)
print("The Homography Matrices H1 and H2 are:")
print(H1,end="\n")
print(H2)

#Drawing Epilines before rectification and finding new homography points using rectification
image1_copy = image1.copy()
image2_copy = image2.copy()
lines1 = cv2.computeCorrespondEpilines(points_left.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
left_img1,left_img2 = draw_epilines(image1_copy,image2_copy,points_left,points_right,lines1)
cv2.imwrite("chess/LeftEpilineBefore.jpg",left_img1)
image1_copy = image1.copy()
image2_copy = image2.copy()
lines2 = cv2.computeCorrespondEpilines(points_right.reshape(-1, 1, 2), 2, F)
lines2 = lines2.reshape(-1, 3)
right_img1,right_img2 = draw_epilines(image2_copy,image1_copy,points_right,points_left,lines2)
cv2.imwrite("chess/RightEpilineBefore.jpg",right_img1)
epi_img = np.concatenate((left_img1,right_img1),axis=1)
cv2.imwrite("chess/InitialEpiline.jpg",epi_img)


#Drawing Epilines after rectification and finding new homography points using rectification
#calculation of rectified points for epipole lines
rectified_left_points,rectified_right_points = get_rectified_new_points(H1,H2,points_left,points_right)
#Getting epipole lines by first applying Warp perspective on the images with Homography matrices
image1_copy = image1.copy()
image2_copy = image2.copy()
image1_rectified = cv2.warpPerspective(image1_copy,H1,(width1,height1))
image2_rectified = cv2.warpPerspective(image2_copy,H2,(width2,height2))
plt.imshow(image1_rectified)
lines1 = cv2.computeCorrespondEpilines(rectified_left_points.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)

left_img1_r,left_img2_r = draw_epilines(image1_rectified,image2_rectified,rectified_left_points,rectified_right_points,lines1)
cv2.imwrite("chess/left_epiline.jpg",left_img1_r)
image1_copy = image1.copy()
image2_copy = image2.copy()
lines2 = cv2.computeCorrespondEpilines(rectified_right_points.reshape(-1, 1, 2), 2, F)
lines2 = lines2.reshape(-1, 3)
right_img1_r,right_img2_r = draw_epilines(image2_rectified,image1_rectified,rectified_right_points,rectified_left_points,lines2)
cv2.imwrite("chess/right_epiline.jpg",right_img1)
epi_img_r = np.concatenate((left_img1_r,right_img1_r),axis=1)
cv2.imwrite("chess/epiline.jpg",epi_img_r)


#Calculating Correspondence
disparity_map = calculate_disparity(image1_rectified,image2_rectified)
disparity_heatmap = cv2.applyColorMap(disparity_map, cv2.COLORMAP_JET)
cv2.imwrite("chess/DisparityMap.jpg",disparity_map)
cv2.imwrite("chess/DisparityHeatMap.jpg",disparity_heatmap)
