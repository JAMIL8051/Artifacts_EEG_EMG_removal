import math 
import numpy as np


#Dot product of two vectors
def dotproduct(v1,v2):
    return sum((a*b) for a,b in zip(v1,v2))


#To get the length(magnitude) of a vector   
def length(v):
    return math.sqrt(dotproduct(v,v))


#orthogonal project of each maps of a group
def orthogonal_projection_3d(data):
    for j in range(0,data.shape[0]):
        nd = data.shape[1]
        while (nd != 3):
            nd = nd-1
            last_element = data[j][nd] # Setting the last element to 1
            t = 1/last_element
            data_scaled = np.empty((data.shape[0],nd),dtype = float, order ='F')
            for i in range(0, nd):
                data_scaled[j][i] = data[j][i]/t
    return data_scaled


#Orthogonal projection in 3d for each map
def orthogonal_projection_3d_map(data):
    nd = len(data)
    
    while (nd != 3):
        nd = nd-1
        last_element = data[nd] # Setting the last element to 1
        t = 1/last_element
        data_scaled = np.empty((nd,1),dtype = float, order ='F')
        for i in range(0, nd):
            data_scaled[i] = data[i]/t
        
    return data_scaled


# Topographic dissimilarity
def topo_dissimilarity(u,v):
    norm_u = normalized_vector(u)
    norm_v = normalized_vector(v)
    dissimilarity = math.sqrt(sum(((a-b)**2) for a,b in zip(norm_u, norm_v)))
    return dissimilarity


# Topographic correlation
def topographic_correlation(v1,v2):
    corr = (dotproduct(v1,v2))/(length(v1)*length(v2))
    return corr


#Function to get unit vector
def normalized_vector(u):
    norm_u =u/length(u)
    return norm_u


def testTopographyConsistancy(data1, data2):
    #This function test the topographic consistency among the maps of the two groups:accross the two groups
    
    data = np.concatenate((data1,data2), axis = 0)# Here axis=0/1 will depend whether we shuffle potential 
    #values of same electrode/channel or just reshuffle electordes. If potential values then axis=0 else 1 
    #for reshuffle of electrodes/channels.
    
    observed_grand_mean = np.mean(data, axis = 0)
    gfp_observed_grand_mean = np.std(observed_grand_mean)
    gfp_after_shuffle = np.zeros(5000)

    for i in range(5000):
        np.random.shuffle(data1.T)# Here data1/data1.T will depend whether we shuffle potential values of 
        #same electrode/channel or just reshuffle electordes. 
        #If potential values then data1 else data1.T for reshuffle of electrodes/channels.
        np.random.shuffle(data2.T)
        
        grand_mean_after_shuffle = np.mean(np.concatenate((data1.T,data2.T),axis = 0))
        gfp_after_shuffle[i] = np.std(grand_mean_after_shuffle)
        
    count = 0

    for i in range(len(gfp_after_shuffle)):
        if gfp_after_shuffle[i] >= gfp_observed_grand_mean:
            count = count+1

    percentage = (count/5000)
    print('\nThe probability of null hypothesis for topographic consistency is :{}'.format(percentage))
    return gfp_after_shuffle


def comparisonMapDiffBetweenTwoConditions(data1,data2):
    mean_data1 = np.mean(data1,axis = 0,keepdims = True)
    mean_data2 = np.mean(data2,axis = 0,keepdims = True)
    diff_map = mean_data1 - mean_data2
    #norm_mean_data1 = normalized_vector(mean_data1)
    #norm_mean_data2 = normalized_vector(mean_data1)
    #norm_diff_map = norm_mean_data1 - norm_mean_data2
    gfp_diff_map = np.std(diff_map)
    #gmd = np.std(norm_diff_map) # Global map dissimilarity
    rand_gfp_diff_map =np.zeros(5000)
    #Concatenate data
    data = np.concatenate((data1, data2), axis = 0)
    for i in range(5000):
        #dataA = np.random.permutation(data1.T)
        #dataB = np.random.permutation(data2.T)

        np.random.shuffle(data)
        dataA  = data[0:data1.shape[0], 0:data1.shape[1]]
        dataB  = data[data1.shape[0]:2*data1.shape[0], 0:data1.shape[1]]

        #lensA = np.array(map(len,dataA)) # Thanks to @Kasramvd on this!
        #valsA = np.concatenate(dataA)
        #shift_idxA = np.append(0,lensA[:-1].cumsum())
        #mean_dataA = np.add.reduceat(valsA,shift_idxA)/lensA.astype(float)
        
        #lensB = np.array(map(len,dataB)) # Thanks to @Kasramvd on this!
        #valsB = np.concatenate(dataB)
        #shift_idxB = np.append(0,lensB[:-1].cumsum())
        #mean_dataB = np.add.reduceat(valsB,shift_idxB)/lensB.astype(float)
        
        
        mean_dataA = np.mean(dataA, axis = 0, keepdims = True)
        mean_dataB = np.mean(dataB, axis = 0, keepdims = True)
        rand_diff_map = mean_dataA - mean_dataB
        rand_gfp_diff_map[i] = np.std(rand_diff_map)
    #print(rand_gfp_diff_map)
    
    count_map_diff_two = 0
    for i in range(0,len(rand_gfp_diff_map)):
        if rand_gfp_diff_map[i] >= gfp_diff_map:
            count_map_diff_two = count_map_diff_two + 1

    probability = count_map_diff_two/len(rand_gfp_diff_map)
    print(gfp_diff_map)
    print('\nThe probability that the "difference between two conditions/groups are observed by: chance" or the probability of NULL hyothesis is: {}'.format(probability))
    
    return rand_gfp_diff_map


#Formatting the data into conditions and subjects
def format_data(data, condition, n_subject, n_ch, ith_class):
        n_condition = len(condition)
        data_temp = np.zeros((n_subject, n_condition, n_ch))
        for i in range(n_subject):
            for j in range(n_condition):
                data_temp[i,j] = data[condition[j]][i][ith_class]
        return data_temp


#Shuffling the indices of the data matrix for easy and faster shuffle of the data matrix
def shuffle_data(data, n_condition):
    for i in range(data.shape[0]):
        random_index = np.random.permutation(n_condition)
        data[i] = data[i][random_index]
    return data


def comparisonMapDiffAcrossConditions(data, condition, n_subject, n_ch, ith_class):
    data_temp = format_data(data, condition, n_subject, n_ch, ith_class) 
    data = data_temp    
    grand_mean_across_subjects = np.mean(data, axis=0)
    grand_mean_across_subjects_across_conditions = np.mean(data, axis=(0, 1))

    residual = np.power(grand_mean_across_subjects - grand_mean_across_subjects_across_conditions, 2)
    generalized_dissimilarity = np.sum(np.sqrt(np.mean(residual, axis=1)))
    
    residual_maps = grand_mean_across_subjects - grand_mean_across_subjects_across_conditions
    #Within-subject factor:Shuffle accross subjects:
    rand_effect_size = np.zeros(5000)
    for i in range(5000):
        data_s = shuffle_data(residual_maps, n_condition = 6)
        rand_grand_mean_across_subjects = np.mean(data_s, axis=0)
        rand_grand_mean_across_subjects_across_conditions = np.mean(data_s, axis=(0, 1))
        residual = np.power(grand_mean_across_subjects - grand_mean_across_subjects_across_conditions, 2)
        rand_generalized_dissimilarity = np.sum(np.sqrt(np.mean(residual, axis=1)))
        rand_effect_size[i] = rand_generalized_dissimilarity

    count_map_diff_multi = 0

    for i in range(0,len(rand_effect_size)):
        if rand_effect_size[i] >= observed_effect_size:
            count_map_diff_multi = count_map_diff_multi + 1

    percentage = count_map_diff_multi/5000
    print('\nThe probability of null hypothesis across different conditions is: {}'.format(percentage))
    return rand_effect_size


def effectsConditions(data1,data2,data3):
    #This function shows: a) The effect for selection of primary channels as group 
    #b) Effect for right group and left group channels to see whether these are of primary channels group 
    #c) Effect of interaction of these two factors: Primary channels group, right group and 
    #left group2 channels
    
    data = np.concatenate((data1,data2,data3), axis = 0)
    grand_mean = np.mean(data,axis = 0)

    #Step 2 page 178 Chap 8 Electrical Neuroimaging book 
    res_maps1 = data1 - grand_mean
    res_maps2 = data2 - grand_mean
    res_maps3 = data3 - grand_mean
     
    #Step 3a.
    grand_mean_res_map1 = np.mean(res_maps1, axis = 0)
    grand_mean_res_map2 = np.mean(res_maps2, axis = 0)
    grand_mean_res_map3 = np.mean(res_maps3, axis = 0)

    #Step 3b.
    obs_effect_size =np.std(grand_mean_res_map1)+ np.std(grand_mean_res_map3)+ np.std(grand_mean_res_map3)
    
    #Step 3c. page 178 Chap 8 Electrical Neuroimaging book
    res_data1 = data1- grand_mean_res_map1
    res_data2 = data2- grand_mean_res_map2
    res_data3 = data3- grand_mean_res_map3

    rand_effect_size = np.zeros(5000)
    rand_effect_size1 = np.zeros(5000)
    rand_effect_size2 = np.zeros(5000)
    
    rand_res_data = np.concatenate((res_maps1,res_maps2,res_maps3),axis = 0)
    for i in range(5000):
        np.random.shuffle(rand_res_data)
        rand_grand_mean_res_map1 = np.mean(rand_res_data[0:data1.shape[0],:], axis = 0)
        rand_grand_mean_res_map2 = np.mean(rand_res_data[data1.shape[0]:2*data1.shape[0],:], axis = 0)
        rand_grand_mean_res_map3 = np.mean(rand_res_data[2*data1.shape[0]:3*data1.shape[0],:], axis = 0)
        
        rand_effect_size[i] = np.std(rand_grand_mean_res_map1)
        rand_effect_size1[i] = np.std(rand_grand_mean_res_map3)
        rand_effect_size2[i] = np.std(rand_grand_mean_res_map3) 
        #Step 3 repeat
        rand_res_data1 = data1- rand_grand_mean_res_map1
        rand_res_data2 = data2- rand_grand_mean_res_map2
        rand_res_data3 = data3- rand_grand_mean_res_map3

    count_con1 = 0
    for i in range(0,len(rand_effect_size)):
        if rand_effect_size[i]>=obs_effect_size:
            count_con1 = count_con1+1
    percentage = count_con1/5000
    print('\nThe probability of the null hypothesis for primary channels group factor:{}'.format(count_con1))
    
    count_con2 = 0
    for i in range(0,len(rand_effect_size)):
        if rand_effect_size1[i]>=obs_effect_size:
            count_con2 = count_con2 +1
    percentage1 = count_con2/5000
    print('\nThe probability of the null hypothesis for channels group left factor:{}'.format(count_con2))
    
    count_con3 = 0
    for i in range(0,len(rand_effect_size)):
        if rand_effect_size2[i]>=obs_effect_size:
            count_con3  = count_con3 +1
    percentage2 = count_con3/5000
    print('\nThe probability of the null hypothesis for channels group right factor:{}'.format(count_con3))

    return percentage, percentage1, percentage2


def freq_effects_test(data1,data2,data3,p=1):
    #this is for the how likely we get the significant results by chance. 
    #Input agrs: data1/data2/dat3: vector(array) of shape no.of randomizations by 1 
    # p = the p-value percentage of threshold. By default 1% 
    first_run1 = data1[0]
    first_run2 = data2[0]
    first_run3 = data3[0]
    
    pseudo_sig1 = 0
    pseudo_sig2 = 0
    pseudo_sig3 = 0

    if p == 5:
        data1 = data1[0:1000]
        data2 = data2[0:1000]
        data3 = data3[0:1000]

    for i in range(1,len(data1)):
        if data1[i]>=first_run1:
            pseudo_sig1=pseudo_sig1+1
    percentage_freq1 = pseudo_sig1/len(data1)
    plt.hist2d(pseudo_sig1, np.linspace(0,5000,10), bins = 10)
    plt.show()

    for i in range(1,len(data2)):
        if data2[i]>=first_run2:
            pseudo_sig1=pseudo_sig1+1
    percentage_freq2 = pseudo_sig1/len(data2)
    plt.hist(pseudo_sig2,np.linspace(0,5000,10), bins = 10)
    plt.show()

    for i in range(1,len(data3)):
        if data1[i]>=first_run1:
            pseudo_sig1=pseudo_sig1+1
    percentage_freq3 = pseudo_sig1/len(data3)
    plt.hist(pseudo_sig3,np.linspace(0,5000,10), bins = 10)
    plt.show()

    return percentage_freq1, percentage_freq2, percentage_freq3




#Spatial analysis function
def spatial_derivative(data_bad, data_grp1):
    data = data_bad
    x = np.linspace(1,len(data), num = len(data))
    data2 = np.empty((len(data),1),dtype = float, order ='F')
    for i in range(len(data)-1):
        data2[i] =data[i+1]-data[i]
    
    data3 = data2
    data4 = np.empty((len(data),1),dtype = float, order='F')
    for i in range(len(data)-1):
        data4[i] = data3[i+1]-data[i]
    
    data_grp1_2 = np.empty((len(data),1),dtype = float, order ='F')
    for i in range(len(data)-1):
        data2[i] =data_grp1[i+1]-data_grp1[i]
    
    data_grp1_3 = data_grp1_2
    
    data_grp1_4 = np.empty((len(data),1),dtype = float, order='F')
    for i in range(len(data)-1):
        data4[i] = data_grp1_3[i+1]-data_grp1_3[i]
    
    return x,data2, data4, data_grp1_2,data_grp1_4
