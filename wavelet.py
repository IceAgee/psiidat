import scipy.fft
import numpy as np

def morlet1DT(new_k,new_w,k_0=6,w_0=6):
    # Computing the wavelet in frequency domain
    out = np.exp( - ((new_k - k_0)**2 + (new_w - w_0)**2)/2.0 )
    return out
def obtain_k_or_w_arr(data_len,freq_sample):
    data_len_2 = int(np.floor((data_len-1)/2))
    array_tmp = np.concatenate((np.arange(0,data_len_2+1,1), np.arange(data_len_2-data_len+1, 0 , 1)))
    k_or_w_arr = 2*np.pi/data_len * array_tmp * freq_sample
    return k_or_w_arr
    
def obtain_new_k_w(k,w,a_s,a_t,c_vel):
    ncvel = np.abs(c_vel)**0.5;
    new_k = a_s * k * ncvel; 
    new_w = a_t * w / ncvel * np.sign(c_vel);
    return new_k, new_w
    
def cwt1DT(fsig,
           freq_sample_spatial,
           freq_sample_time,
           lambda_arr, period_arr,
           c_vel_arr, normalization='L2'):
    """
    Purpose:
    Input: 
        fsig: the 2D fft result of the 1D+1T signal by invoking the function of scipy.fft.fftn;
        freq_sample_spatial: sampling frequency in spatial domain, which is defined as 1/dx;
        freq_sample_time: sampling frequency in time domain, which is defined as 1/dt;
        lambda_arr: the 1d array of wavelength to be analyzed with wavelet transform;
        period_arr: the 1d array of period to be analyzed with wavelet transform;
        c_vel_arr: the 1d array of velocity scaling coefficients to be analyzed with wavelet transform;
        normalization: 'L1' and 'L2'. default: 'L2'
    Output:
        out_data: the 5d array of complexed wavelet coefficient, which involves the dimensions of '1D + 1T + lambda + period + c_vel'
    Record:
        first written by Chuanpeng Hou in 2023-03-14;
    """
    # fsig: fft of signal # 傅立叶变换结果：前一半系数对应正频靠近坐标原点; 后一半对应负频，顺序序频率逐渐降低，如[1,2,3,-3,-2,-1]Hz.
    
    # 检查傅立叶变换后行列关系：是否和原数据保持一致？ Done! 保持一致
    data_len_x = np.shape(fsig)[0]
    data_len_t = np.shape(fsig)[1]
    
    # 检查傅立叶变换后系数和频率的对应关系：是否需要平移频率中心？ Done! 不需要
    k_arr = obtain_k_or_w_arr(data_len_x,freq_sample_spatial) ##是否需要指定采样频率？？需要
    w_arr = obtain_k_or_w_arr(data_len_t,freq_sample_time)
    k_2D_grid,w_2D_grid = np.meshgrid(k_arr,w_arr,indexing='ij')
    
    #创建一个空的复数矩阵
    out_data = np.zeros(shape=(data_len_x,data_len_t,
                               len(lambda_arr),len(period_arr),
                               len(c_vel_arr)), 
                        dtype=complex)
    #循环每个参数
    k_0 = 6
    w_0 = 6
    epsilon = 1
    for lambda_index in range(len(lambda_arr)):
        for period_index in range(len(period_arr)):
            for c_vel_index in range(len(c_vel_arr)):
                    a_s = lambda_arr[lambda_index]/2.0/np.pi*k_0
                    a_t = period_arr[period_index]/2.0/np.pi*w_0
                    c_vel = c_vel_arr[c_vel_index]
                    # k and w come from meshgrid, which should be 3D.
                    # other four parameters are a single scalar. we need four loops.
                    new_k, new_w = obtain_new_k_w(k_2D_grid, w_2D_grid, a_s, a_t, c_vel)
                    # Call of the wavelet function.
                    mask = morlet1DT(new_k, new_w, k_0=k_0, w_0=w_0)
                    if normalization == 'L1':
                        out_data[:,:,lambda_index,period_index,c_vel_index] = scipy.fft.ifftn(fsig * np.conj(mask), axes=(0,1))#检查是否是矩阵元素相乘
                    if normalization == 'L2':
                        out_data[:,:,lambda_index,period_index,c_vel_index] = np.abs(a_s)**(0.5) * np.abs(a_t)**(0.5) * scipy.fft.ifftn(fsig * np.conj(mask), axes=(0,1))#检查是否是矩阵元素相乘
    return out_data



# 对扰动的1D+T的小波变换结果S_wavelet_out，获得指定位置和时间上的扰动delta_variable(x,t)，即，作逆变换。
def morlet1DT_xt_domain(new_x,new_t,k_0=6,w_0=6):
    # Computing the wavelet in x-t domain
    # reference: 小波函数的选取【金山文档】 研究日志—二维全球模拟磁流体力学波动传播 https://kdocs.cn/l/chpvOJkGNnfR
    first_item = np.exp(-1/2.0 * new_t**2)
    second_item = np.exp(complex(0,1) * w_0 * new_t) - np.exp(-1/2.0 * w_0**2)
    third_item = np.exp(-1/2.0 * new_x**2)
    fourth_item = np.exp(complex(0,1) * k_0 * new_x) - np.exp(-1/2.0 * k_0**2)
    out = first_item * second_item * third_item * fourth_item
    return out
def obtain_new_x_t(x,t, bx_5D_grid, tau_5D_grid, a_s_5D_grid, a_t_5D_grid, c_vel_5D_grid):
    new_x = a_s_5D_grid**(-1.0) * c_vel_5D_grid**(-1/2.0) * (x - bx_5D_grid) 
    new_t = a_t_5D_grid**(-1.0) * c_vel_5D_grid**( 1/2.0) * (t - tau_5D_grid)
    return new_x,new_t
def icwt1DT(S_wavelet_out, x_arr, t_arr,
            bx_arr, tau_arr,
            lambda_arr, period_arr, c_vel_arr, k_0=6, w_0=6):
    """
    Purpose:
    Input: 
        S_wavelet_out: the wavelet of the 1D+1T signal by invoking the function of cwt1DT;
        x_arr: the sptial position  ;
        t_arr: sampling frequency in time domain, which is defined as 1/dt;
        lambda_arr: the 1d array of wavelength to be analyzed with wavelet transform;
        period_arr: the 1d array of period to be analyzed with wavelet transform;
        c_vel_arr: the 1d array of velocity scaling coefficients to be analyzed with wavelet transform;
        normalization: 'L1' and 'L2'. default: 'L2'
    Output:
        out_data: the 5d array of complexed wavelet coefficient, which involves the dimensions of '1D + 1T + lambda + period + c_vel'
    Record:
        first written by Chuanpeng Hou in 2023-03-14;
    """
    #
    d_bx = np.nanmean(np.diff(bx_arr))
    d_tau = np.nanmean(np.diff(tau_arr))
    d_as = np.nanmean(np.diff(lambda_arr))/2.0/np.pi*k_0
    d_at = np.nanmean(np.diff(period_arr))/2.0/np.pi*w_0
    d_c = np.nanmean(np.diff(c_vel_arr))
    # 
    d_bx  = 1 if not np.isinf(d_bx)  else d_bx
    d_tau = 1 if not np.isinf(d_tau) else d_tau
    d_as  = 1 if not np.isinf(d_as)  else d_as
    d_at  = 1 if not np.isinf(d_at)  else d_at
    d_c   = 1 if not np.isinf(d_c)   else d_c
    #
    a_s_arr = lambda_arr/2.0/np.pi*k_0
    a_t_arr = period_arr/2.0/np.pi*w_0
    bx_5D_grid, tau_5D_grid, a_s_5D_grid, a_t_5D_grid, c_vel_5D_grid = np.meshgrid(bx_arr,tau_arr,a_s_arr,a_t_arr,c_vel_arr,indexing='ij')
    out_data = np.zeros(shape=(len(x_arr),len(t_arr)), dtype=complex)
    for x_index in range(len(x_arr)):
        for t_index in range(len(t_arr)):
            x = x_arr[x_index]
            t = t_arr[t_index]
            new_x_5D, new_t_5D = obtain_new_x_t(x, t, bx_5D_grid, tau_5D_grid, a_s_5D_grid, a_t_5D_grid, c_vel_5D_grid)
            # Call of the wavelet function.
            mask_5D = morlet1DT_xt_domain(new_x_5D,new_t_5D,k_0=6,w_0=6)
            out_data[x_index, t_index] = np.nansum(mask_5D * S_wavelet_out) * d_bx * d_tau * d_as * d_at * d_c #对积分号内的值求和
    return np.real(out_data) # 取实部作为反变换信号

