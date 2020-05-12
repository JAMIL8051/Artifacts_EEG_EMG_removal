#Formation of Data matrix for Partial Least Square
def concatenate_data_by_condition(data, task_condition, exclude_task):
    res = OrderedDict()
    for condition in task_condition:
        for task_name, task_data in data.items():
            if task_name not in exclude_task:
                if condition == task_name.split("_")[1]:
                    res[condition] = np.asarray(task_data) if condition not in res else np.concatenate((res[condition], np.asarray(task_data)), axis=1)
    return res


def pls(data, task_condition, exclude_task, condition, n_subject,n_ch):
    """ Spatio-temporal Partial least square: As detailed in the reference. 
        [1] Partial least squares analysis of neuroimaging data: applications and advances.
        By Anthony Randal McIntosha,* and Nancy J. Lobaughb 
        aRotman Research Institute of Baycrest Centre, University of Toronto, Toronto, Ontario, Canada M6A 2E1
        bSunnybrook and Women’s College Health Sciences Centre, 
        University of Toronto, Toronto, Ontario, Canada M4N 3M5

        Args: 
            data: numpy.array, size = no. of time samples (time_samples) by number of EEG channels (n_ch)
        
    """ 
    res = concatenate_data_by_condition(data, task_condition, exclude_task)
    #No. of time samples. data parameter should have all the time samples as rows and n_ch as columns 
 
    time_samples = data.shape[0]#Ideally the data parameter should contain the no. of time samples
    n_condition = len(condition)
    data_temp = np.zeros((n_condition, n_subject, t*n_ch))
    for i in range(n_condition):
        for j in range(n_subject):
            data_temp[i,j] = res[condition].flatten('F')
    

    #Design_matrix:Contrast matrix
    design_mat = np.zeros((n_subject,n_condition,n_condition-1),dtype ='int') 
    for subject in range(n_subject):
        val = np.arange(n_condition,0,-1)
        val = np.diag(val)
        temp = (1)*np.ones((n_condition,n_condition),dtype='int')
        val = val + temp
        val = val[:,0:n_condition-1]
        contrast_mat = np.tril(val, k=0)
        design_mat[subject,:,:] = contrast_mat

    design_mat = design_mat.reshape((n_subject*n_condition,n_condition-1))

    #Data matrix:It should contain n*k rows where n=no.of subject and k = no.of conditions to compare
    #and each row of the matrix will contain t*n_ch columns 
    #where t=no.of time samples and n_ch=no.of channels
     
    M = data_temp.reshape((n_subject*n_condition,time_samples*n_ch))
    vec_ones = np.ones((n_subjects*n_condition),dtype ='int')
    I = vec_ones
    C_trans = design_mat.T 
    part = M - np.dot(I,(np.dot(I.T,M)/n_subject*n_condition))
    part = part/(n_condition-1)
    Y = np.dot(C_trans, part)

#Application of Singular value decomposition(SVD)
    u,s,vh = np.linalg.svd(Y.T,full_matrices = False)
    obs_effect_size = s
    rand_effect_size = np.zeros((5000,len(s)),dtype = 'float')
    for i in range(5000):
        rand_des_mat = shuffle_data(design_mat, n_condition)
        rand_C_trans = rand_des_mat.T
        rand_part = M - np.dot(I,(np.dot(I.T,M)/n_subject*n_condition))
        rand_part = rand_part/(n_condition-1)
        rand_Y = np.dot(rand_C_trans,rand_part)
        rand_u,rand_s,rand_vh = np.linalg.svd(rand_Y.T,full_matrices = False, compute_uv = True)
        rand_effect_size[i] = rand_s

    count_LV1 = 0
    count_LV2 = 0
    rand_effect_size_mod1 = rand_effect_size[:,0]
    rand_effect_size_mod2 = rand_effect_size[:,1]
    for i in range(0,len(rand_effect_size)):
        if(rand_effect_size_mod1[i]>=obs_effect_size[0]):
            count_LV1 += 1
        if(rand_effect_size_mod2[i]>=obs_effect_size[0]):
            count_LV2 += 1

    p_value_LV1 = count_LV1/5000
    p_value_LV2 = count_LV2/5000

    return p_value_LV1, p_value_LV2